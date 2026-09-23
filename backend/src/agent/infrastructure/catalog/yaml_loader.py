import json
from pathlib import Path
import warnings

from ...domain.routing.entities import ModelCatalog
from ...domain.routing.value_objects import ModelSpec, ModelTier


DEFAULT_CATALOG_JSON = '''{"models": [
  {"name": "qwen2.5-coder:1.5b", "params_b": 1.5, "size_gb": 1.1, "tier": "executor", "context_window": 32768},
  {"name": "qwen2.5-coder:3b", "params_b": 3, "size_gb": 2.2, "tier": "executor", "context_window": 32768},
  {"name": "qwen2.5-coder:7b", "params_b": 7, "size_gb": 4.8, "tier": "planner", "context_window": 32768},
  {"name": "qwen2.5-coder:14b", "params_b": 14, "size_gb": 9.5, "tier": "planner", "context_window": 32768}
]}'''


def load_catalog(path: Path) -> ModelCatalog:
    try:
        import yaml
    except ImportError:
        warnings.warn("PyYAML ausente: usando catalogo embutido; arquivo informado nao sera lido.",
                      RuntimeWarning, stacklevel=2)
        data = json.loads(DEFAULT_CATALOG_JSON)
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("Catalog must contain a models list")
    models: list[ModelSpec] = []
    for item in data["models"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid catalog entry")
        models.append(ModelSpec(
            name=item["name"], params_b=item["params_b"], size_gb=item["size_gb"],
            tier=ModelTier(item["tier"]), context_window=item["context_window"],
        ))
    return ModelCatalog(tuple(models))
