import chromadb
import os
import nest_asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, ExactMatchFilter
from llama_index.core.schema import TextNode

# Load environment variables
load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

# Apply nest_asyncio for async operations in notebooks/scripts
nest_asyncio.apply()

class HybridVectorDB:
    def __init__(self, user_id: str, base_db_path: str = r"C:\My Projects\Health-Navigator\app\workflow\vectordb\chroma_db", 
                 google_api_key: str = None, model_name: str = "models/embedding-001"):
        """
        Initialize a user-isolated ChromaDB with Google embeddings.
        Each user gets their own folder to prevent database locking/corruption.
        """
        
        # 1. User Isolation: Create a specific folder for this user
        self.user_id = user_id
        self.db_path = os.path.join(base_db_path, f"user_{user_id}")
        
        # Ensure the directory exists
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

        # 2. Setup Google Models (Passed explicitly to avoid global Settings conflicts)
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required. Pass google_api_key or set GOOGLE_API_KEY env variable")
        
        self.llm = GoogleGenAI(model="gemini-2.5-flash-lite-preview-09-2025", api_key=api_key)
        self.embed_model = GoogleGenAIEmbedding(model_name=model_name, api_key=api_key)
        
        # 3. Connect to ChromaDB (Persistent)
        # Using a persistent client on the user's specific folder
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Create/Get collection (Name can be simple since the path is already unique)
        self.collection = self.client.get_or_create_collection(
            name="user_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 4. Setup LlamaIndex Vector Store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize Index with specific embed_model
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model,
            storage_context=self.storage_context
        )

        self.parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separator="\n\n"
        )

        # 5. Load Nodes for BM25
        # We load directly from the collection to ensure BM25 has data to work with
        self.all_nodes = self._load_all_nodes()

    def _load_all_nodes(self) -> List[TextNode]:
        """Load all nodes directly from ChromaDB collection for BM25 initialization."""
        try:
            # Get all data from collection
            results = self.collection.get(include=['documents', 'metadatas'])
            nodes = []
            if results and results['ids']:
                for i in range(len(results['ids'])):
                    node = TextNode(
                        text=results['documents'][i],
                        metadata=results['metadatas'][i] or {},
                        id_=results['ids'][i]
                    )
                    nodes.append(node)
            return nodes
        except Exception as e:
            print(f"Error loading nodes: {e}")
            return []

    def _date_to_int(self, date_str) -> int:
        """Convert date string 'YYYY-MM-DD' to integer YYYYMMDD."""
        if not date_str: return 0
        if isinstance(date_str, int): return date_str
        return int(date_str.replace("-", ""))
    
    def add_text(self, text: str, metadata: Dict = None) -> bool:
        """Add text to the user's specific database."""
        metadata = metadata or {}

        # Convert date to int for filtering
        if "date" in metadata:
            metadata["date"] = self._date_to_int(metadata["date"])

        try:
            doc = Document(text=text, metadata=metadata)
            nodes = self.parser.get_nodes_from_documents([doc])
            
            # Insert into Vector Index
            self.index.insert_nodes(nodes)
            
            # Update local list for BM25 immediately
            self.all_nodes.extend(nodes)
            
            return True
        except Exception as e:
            print(f"Error adding text: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 5, initial_k: int = None, 
             similarity_threshold: float = 0.0, filters: Dict = None,
             date: str = None, date_filter: str = None):
        """
        Retrieve relevant texts using hybrid search (semantic + BM25) with filters.
        """
        try:
            initial_k = initial_k or (top_k * 3)
            
            # --- 1. Construct Metadata Filters ---
            filter_list = []
            
            # Add strict key-value filters
            if filters:
                filter_list.extend([
                    ExactMatchFilter(key=k, value=v) for k, v in filters.items()
                ])
            
            # Add date filter
            if date and date_filter:
                operator_map = {
                    "before": FilterOperator.LTE,
                    "at": FilterOperator.EQ,
                    "after": FilterOperator.GTE
                }
                if date_filter in operator_map:
                    filter_list.append(
                        MetadataFilter(
                            key="date", 
                            value=self._date_to_int(date), 
                            operator=operator_map[date_filter]
                        )
                    )
            
            metadata_filters = MetadataFilters(filters=filter_list) if filter_list else None

            # --- 2. Vector Retriever ---
            vector_retriever = self.index.as_retriever(
                similarity_top_k=initial_k,
                filters=metadata_filters,
                embed_model=self.embed_model 
            )

            # --- 3. BM25 Retriever (with Python-side filtering) ---
            # BM25 doesn't support vector store filters natively, so we filter the nodes list first
            filtered_nodes_bm25 = self.all_nodes
            
            if filters or (date and date_filter):
                date_int = self._date_to_int(date) if date else 0
                filtered_nodes_bm25 = []
                for node in self.all_nodes:
                    # Check exact match filters
                    match = True
                    if filters:
                        if not all(node.metadata.get(k) == v for k, v in filters.items()):
                            match = False
                    
                    # Check date filters
                    if match and date and date_filter:
                        node_date = node.metadata.get("date")
                        if not node_date:
                            match = False
                        else:
                            if date_filter == "before" and not (node_date <= date_int): match = False
                            elif date_filter == "at" and not (node_date == date_int): match = False
                            elif date_filter == "after" and not (node_date >= date_int): match = False
                    
                    if match:
                        filtered_nodes_bm25.append(node)
            
            # If no nodes match filters, fall back to vector search only to avoid BM25 crash
            if not filtered_nodes_bm25:
                # If we have no nodes at all, return empty
                if not self.all_nodes: 
                    return []
                # If we just have no matching nodes for BM25, the vector retriever might still find something 
                # (though unlikely if filters match), but we proceed with just vector retriever
                retrieved_nodes = vector_retriever.retrieve(query)
            else:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=filtered_nodes_bm25,
                    similarity_top_k=initial_k
                )

                # --- 4. Hybrid Fusion ---
                hybrid_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=top_k,
                    num_queries=1, # Keep at 1 for speed, increase to 3 for better accuracy (multi-query)
                    mode="reciprocal_rerank",
                    use_async=True,
                    llm=self.llm
                )
                
                retrieved_nodes = hybrid_retriever.retrieve(query)

            # --- 5. Output Formatting ---
            results = []
            for node in retrieved_nodes:
                # RRF scores are usually small (0 to 1), check threshold carefully
                if hasattr(node, 'score') and node.score is not None:
                     if node.score >= similarity_threshold:
                        results.append({
                            "text": node.node.get_content(),
                            "metadata": node.node.metadata,
                            "score": node.score
                        })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving: {e}")
            import traceback
            traceback.print_exc()
            return []