from dataclasses import dataclass

_BASE_BLOBS_LOCATION = "/usr/share/ollama/.ollama/models/blobs"

LLAMA_CPP_MODEL_PATHS = {
    "qwen3:14b": f"{_BASE_BLOBS_LOCATION}/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e",
    "qwen3:8b": f"{_BASE_BLOBS_LOCATION}/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
}

@dataclass
class LLMProvider():
    """Represents a single LLM provider configuration."""
    provider: str
    model_name: str
    
    def __getitem__(self, key):
        return getattr(self, key)
    

@dataclass
class LLMSettings(LLMProvider):
    """LLM selection and runtime settings."""    
    temperature: float
    context_window: int | None = None  # only for ollama
    top_p: float | None = 1  # only for ollama for now
    no_thinking: bool = False  # disable reasoning for supported models

    
    def __getitem__(self, key):
        return getattr(self, key)