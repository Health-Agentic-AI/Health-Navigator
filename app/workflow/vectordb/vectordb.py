import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from typing import List, Dict
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.vector_stores import MetadataFilter, FilterOperator

from dotenv import load_dotenv
import os

load_dotenv(r'C:\My Projects\Health-Navigator\credentials.env')

import nest_asyncio
nest_asyncio.apply()


class HybridVectorDB:
    def __init__(self, user_id: str, db_path: str = r"C:\My Projects\Health-Navigator\app\workflow\vectordb\chroma_db", google_api_key: str = None, model_name: str = "models/embedding-001"):
        """Initialize connection to existing ChromaDB with Google embeddings."""

        self.all_nodes = []
        self.db_path = db_path
        self.user_id = user_id
        self.collection_name = f"user_{user_id}"
        
        # Setup Google embeddings
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required. Pass google_api_key or set GOOGLE_API_KEY env variable")
        
        self.llm = GoogleGenAI(model="gemini-2.5-flash-lite-preview-09-2025", api_key=api_key)
        
        self.embed_model = GoogleGenAIEmbedding(
            model_name=model_name,
            api_key=api_key
        )
        Settings.embed_model = self.embed_model
        
        # Verify DB exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB path '{db_path}' does not exist")
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
                configuration={
            "hnsw": {"space": "cosine"},
        }
            )
        
        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Load index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model
        )

        self.parser = SentenceSplitter(
        chunk_size=1024,  # Larger for medical context
        chunk_overlap=200,
        separator="\n\n"  # Split by paragraphs/sections
        )

        try:
            retriever = self.index.as_retriever(similarity_top_k=100000)
            self.all_nodes = retriever.retrieve("dummy query to load all")
            self.all_nodes = [node.node for node in self.all_nodes]
        except:
            self.all_nodes = []

    def _load_all_nodes(self):
        """Load all nodes from ChromaDB collection."""
        from llama_index.core.schema import TextNode
        try:
            results = self.collection.get(include=['documents', 'metadatas'])
            seen_texts = set()
            nodes = []
            for i in range(len(results['ids'])):
                text = results['documents'][i]
                if text not in seen_texts:
                    seen_texts.add(text)
                    node = TextNode(
                        text=text,
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
        if isinstance(date_str, int):
            return date_str
        return int(date_str.replace("-", ""))
    
    def add_text(self, text: str, metadata: Dict = None) -> bool:
        """
        Add text to the database.
        
        Args:
            text: String content to add
            metadata: Optional metadata dictionary
            
        Returns:
            bool: True if successful
        """

        metadata = metadata or {}

        # Before inserting, convert date to int if present
        if metadata and "date" in metadata:
            metadata["date"] = self._date_to_int(metadata["date"])

        try:
            # Create document
            doc = Document(text=text, metadata=metadata or {})
            
            # Parse into nodes
            nodes = self.parser.get_nodes_from_documents([doc])
            
            # Insert into index
            self.index.insert_nodes(nodes)
            
            # Store for BM25
            self.all_nodes.extend(nodes)
            
            return True
        except Exception as e:
            print(f"Error adding text: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 100, initial_k: int = None, 
             similarity_threshold = 0.01, filters: Dict = None,
             date: str = None, date_filter: str = None):
        """
        Retrieve relevant texts using hybrid search (semantic + BM25).
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved text strings
        """

        
        try:
            # Get all nodes for BM25

            initial_k = initial_k or max(top_k * 3, 30)
            all_nodes = self.all_nodes
            
            if not all_nodes:
                return []
            
            metadata_filters = None
            if filters or date:
                filter_list = []
                
                # Add existing filters
                if filters:
                    filter_list.extend([
                        ExactMatchFilter(key=k, value=v) 
                        for k, v in filters.items()
                    ])
                
                # Add date filter
                if date and date_filter:
                    operator_map = {
                        "before": FilterOperator.LTE,
                        "at": FilterOperator.EQ,
                        "after": FilterOperator.GTE
                    }
                    if date_filter not in operator_map:
                        raise ValueError("date_filter must be 'before', 'at', or 'after'")
                    
                    filter_list.append(
                        MetadataFilter(key="date", value=self._date_to_int(date), operator=operator_map[date_filter])
                    )
                
                if filter_list:
                    metadata_filters = MetadataFilters(filters=filter_list)
            
            # Semantic retriever
            vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=initial_k,
                filters=metadata_filters
            )

            filtered_nodes = all_nodes
            if filters or date:
                filtered_nodes = [
                    node for node in all_nodes 
                    if all(node.metadata.get(k) == v for k, v in (filters or {}).items())
                ]

                # Apply date filter for BM25
                if date and date_filter:
                    date_int = self._date_to_int(date)
                    filtered_nodes = [
                        node for node in filtered_nodes
                        if node.metadata.get("date") and (
                            (date_filter == "before" and self._date_to_int(node.metadata.get("date")) <= date_int) or
                            (date_filter == "at" and self._date_to_int(node.metadata.get("date")) == date_int) or
                            (date_filter == "after" and self._date_to_int(node.metadata.get("date")) >= date_int)
                        )
                    ]

            # Check if we have nodes before creating BM25 retriever
            if not filtered_nodes:
                # No matching nodes for BM25, use only vector retriever
                retrieved_nodes = vector_retriever.retrieve(query)
            else:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=filtered_nodes,
                    similarity_top_k=initial_k,
                )
                
                # Hybrid retriever
                hybrid_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=initial_k,
                    num_queries=3,
                    mode="reciprocal_rerank",
                    use_async=True,
                    llm=self.llm
                )
                
                # Retrieve
                retrieved_nodes = hybrid_retriever.retrieve(query)

            # Filter by similarity threshold

            retrieved_nodes = [node for node in retrieved_nodes if hasattr(node, 'score') and node.score is not None and node.score >= similarity_threshold]
            
            # Extract text
            results = [
                {"text": node.text, "metadata": node.metadata, "score": node.score}
                for node in retrieved_nodes[:top_k]
            ]
            return results
            
        except Exception as e:
            print(f"Error retrieving: {e}")
            import traceback
            traceback.print_exc()
            return []
