"""
Health Navigator - Model Registry
Centralized model loading, unloading, and management
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered ML model."""
    name: str
    version: str
    model_type: str  # 'classification', 'regression', 'vision', 'nlp', etc.
    file_path: str
    accuracy: Optional[float] = None
    description: Optional[str] = None
    training_date: Optional[datetime] = None
    parameters: Optional[int] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    loaded: bool = False
    load_time_ms: Optional[float] = None
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type,
            'file_path': self.file_path,
            'accuracy': self.accuracy,
            'description': self.description,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'parameters': self.parameters,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'loaded': self.loaded,
            'load_time_ms': self.load_time_ms,
            'last_used': self.last_used.isoformat() if self.last_used else None,
        }


class ModelRegistry:
    """
    Centralized registry for managing ML models.

    Handles model loading, unloading, and provides
    a single source of truth for model availability and metadata.
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize the model registry.

        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models'
        )
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._load_times: Dict[str, float] = {}

        # Ensure models directory exists
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ModelRegistry initialized with models directory: {self.models_dir}")

    def register(
        self,
        name: str,
        loader_func,
        version: str = "1.0.0",
        model_type: str = "unknown",
        accuracy: Optional[float] = None,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        **kwargs
    ) -> ModelMetadata:
        """
        Register a model with the registry.

        Args:
            name: Unique identifier for the model
            loader_func: Function to load the model (called lazily)
            version: Model version string
            model_type: Type of model (classification, regression, etc.)
            accuracy: Model accuracy score (if available)
            description: Human-readable description
            file_path: Path to model file
            **kwargs: Additional metadata fields

        Returns:
            ModelMetadata: The registered model metadata
        """
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model_type,
            file_path=file_path or "",
            accuracy=accuracy,
            description=description,
            **kwargs
        )

        self._metadata[name] = metadata
        self._models[name] = None  # Lazy load

        logger.info(f"Registered model: {name} v{version} ({model_type})")
        return metadata

    def load(self, name: str) -> Any:
        """
        Load a model by name.

        Args:
            name: Model identifier

        Returns:
            The loaded model

        Raises:
            KeyError: If model not registered
            RuntimeError: If model fails to load
        """
        if name not in self._metadata:
            raise KeyError(f"Model '{name}' not registered")

        import time
        start_time = time.time()

        # Check if already loaded
        if self._models.get(name) is not None:
            self._metadata[name].last_used = datetime.now()
            logger.debug(f"Model '{name}' already loaded")
            return self._models[name]

        # Load the model
        try:
            logger.info(f"Loading model: {name}")
            model = self._load_model_impl(name)
            self._models[name] = model
            self._metadata[name].loaded = True

            load_time = (time.time() - start_time) * 1000
            self._metadata[name].load_time_ms = load_time
            self._metadata[name].last_used = datetime.now()

            logger.info(f"Model '{name}' loaded in {load_time:.2f}ms")
            return model

        except Exception as e:
            logger.error(f"Failed to load model '{name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model '{name}': {e}") from e

    def _load_model_impl(self, name: str) -> Any:
        """
        Internal implementation of model loading.

        This method should be overridden or extended based on
        the specific ML framework being used.
        """
        # Try to load from file if path exists
        metadata = self._metadata[name]
        if metadata.file_path and os.path.exists(metadata.file_path):
            return self._load_from_file(metadata.file_path)

        # Otherwise, use framework-specific loading
        return self._load_model_by_type(name, metadata.model_type)

    def _load_from_file(self, file_path: str) -> Any:
        """Load model from file based on extension."""
        ext = Path(file_path).suffix.lower()

        if ext == '.onnx':
            import onnxruntime as ort
            return ort.InferenceSession(file_path)
        elif ext in ('.pt', '.pth'):
            import torch
            return torch.load(file_path, map_location='cpu')
        elif ext == '.pkl':
            import joblib
            return joblib.load(file_path)
        elif ext == '.h5':
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(file_path)
            except ImportError:
                raise RuntimeError("TensorFlow not installed")
        else:
            raise ValueError(f"Unsupported model file format: {ext}")

    def _load_model_by_type(self, name: str, model_type: str) -> Any:
        """Load model based on type using framework-specific code."""
        # Heart disease prediction
        if name == "heart_disease":
            from app.workflow.ml_models.heart_disease_model import predict_heart_disease
            return predict_heart_disease

        # Stroke prediction
        elif name == "stroke":
            from app.workflow.ml_models.stroke_model import predict_stroke
            return predict_stroke

        # Cancer prediction
        elif name == "cancer":
            from app.workflow.ml_models.cancer_model import predict_cancer
            return predict_cancer

        # Medical image analysis
        elif name == "medical_image":
            from app.workflow.ml_models.image_model import analyze_medical_image
            return analyze_medical_image

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def unload(self, name: str) -> None:
        """
        Unload a model to free memory.

        Args:
            name: Model identifier
        """
        if name in self._models:
            del self._models[name]
            self._metadata[name].loaded = False
            logger.info(f"Model '{name}' unloaded")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a model, loading it if necessary.

        Args:
            name: Model identifier

        Returns:
            The model or None if not found
        """
        try:
            return self.load(name)
        except (KeyError, RuntimeError):
            logger.warning(f"Model '{name}' not available")
            return None

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return name in self._models and self._models[name] is not None

    def is_available(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._metadata

    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self._metadata.get(name)

    def list_models(self, loaded_only: bool = False) -> List[str]:
        """
        List all registered models.

        Args:
            loaded_only: If True, only return loaded models

        Returns:
            List of model names
        """
        if loaded_only:
            return [name for name, model in self._models.items() if model is not None]
        return list(self._metadata.keys())

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered models."""
        return {
            name: metadata.to_dict()
            for name, metadata in self._metadata.items()
        }

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        for name in list(self._models.keys()):
            self.unload(name)
        logger.info("All models unloaded")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get estimated memory usage of loaded models.

        Returns:
            Dictionary with memory usage information
        """
        import sys

        usage = {
            'loaded_models': len([m for m in self._models.values() if m is not None]),
            'total_models': len(self._metadata),
            'models': {}
        }

        for name, model in self._models.items():
            if model is not None:
                usage['models'][name] = {
                    'size_bytes': sys.getsizeof(model),
                    'load_time_ms': self._metadata[name].load_time_ms
                }

        return usage

    def warmup(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Pre-load models into memory.

        Args:
            model_names: List of models to warm up. If None, warm up all registered models.

        Returns:
            Dictionary mapping model names to load success status
        """
        if model_names is None:
            model_names = self.list_models()

        results = {}
        for name in model_names:
            try:
                self.load(name)
                results[name] = True
            except Exception as e:
                logger.error(f"Failed to warm up model '{name}': {e}")
                results[name] = False

        return results

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all models.

        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }

        for name, metadata in self._metadata.items():
            model_health = {
                'registered': True,
                'loaded': metadata.loaded,
                'available': self.is_loaded(name)
            }

            if metadata.loaded:
                model_health['last_used'] = metadata.last_used.isoformat() if metadata.last_used else None
                model_health['load_time_ms'] = metadata.load_time_ms

            health['models'][name] = model_health

        # Overall status based on loaded models
        loaded_count = sum(1 for m in health['models'].values() if m['loaded'])
        if loaded_count == 0:
            health['status'] = 'warning'
            health['message'] = 'No models currently loaded'

        return health


# Global model registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        models_dir = os.environ.get('MODELS_DIR')
        _registry = ModelRegistry(models_dir)
    return _registry


def init_default_models() -> None:
    """Initialize and register default models."""
    registry = get_registry()

    # Register heart disease model
    registry.register(
        name="heart_disease",
        loader_func=None,  # Will use default loader
        version="1.0.0",
        model_type="classification",
        accuracy=0.85,
        description="Predicts heart disease risk based on patient vitals and health indicators",
        file_path=os.path.join(registry.models_dir, "heart_disease_model.onnx"),
        tags={"domain": "cardiology", "input": "tabular"}
    )

    # Register stroke model
    registry.register(
        name="stroke",
        loader_func=None,
        version="1.0.0",
        model_type="classification",
        accuracy=0.82,
        description="Predicts stroke risk using demographic and health factors",
        file_path=os.path.join(registry.models_dir, "stroke_model.onnx"),
        tags={"domain": "neurology", "input": "tabular"}
    )

    # Register cancer model
    registry.register(
        name="cancer",
        loader_func=None,
        version="1.0.0",
        model_type="classification",
        accuracy=0.78,
        description="Predicts cancer risk based on symptoms and patient history",
        file_path=os.path.join(registry.models_dir, "cancer_model.onnx"),
        tags={"domain": "oncology", "input": "tabular"}
    )

    # Register medical image model
    registry.register(
        name="medical_image",
        loader_func=None,
        version="1.0.0",
        model_type="vision",
        accuracy=0.75,
        description="Analyzes medical images (X-rays, MRIs, CT scans) for abnormalities",
        file_path=os.path.join(registry.models_dir, "medical_image_model.onnx"),
        tags={"domain": "radiology", "input": "image"}
    )

    logger.info("Default models registered in registry")


__all__ = [
    'ModelRegistry',
    'ModelMetadata',
    'get_registry',
    'init_default_models',
]
