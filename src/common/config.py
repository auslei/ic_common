import logging
import sys
import yaml
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Determine config path relative to the caller (assumed project root)
CONFIG_PATH = Path("config.yaml")

_config_logger = logging.getLogger(__name__)

class LLMConfig(BaseModel):
    provider: str = Field(default="ollama")
    model: str = Field(default="qwen3:14b")
    temperature: float = Field(default=0.0)
    context_window: Optional[int] = Field(default=None)
    concurrency: int = Field(default=2)

class UserConfig(BaseModel):
    """Configuration for a specific user within a core_app instance."""
    id: str
    password: str
    is_admin: bool = Field(default=False)
    session_count: int = Field(default=3)
    collections: List[str] = Field(default_factory=lambda: ["*"])
    allowed_models: List[str] = Field(default_factory=lambda: ["ollama:*"])

class ChatLLMConfig(BaseModel):
    # Support legacy flat structure
    provider: str = Field(default="ollama")
    model: str = Field(default="qwen3:8b")
    temperature: float = Field(default=0.1)
    context_window: int = Field(default=16384)
    concurrency: int = Field(default=2)
    
    # Specialized roles for performance optimization
    router: Optional[LLMConfig] = None
    generator: Optional[LLMConfig] = None

class SummarizationThinkingConfig(BaseModel):
    researcher: bool = Field(default=True)
    synthesizer: bool = Field(default=False)

class SummarizationConfig(BaseModel):
    thinking: SummarizationThinkingConfig = Field(default_factory=SummarizationThinkingConfig)
    templates: Dict[str, str] = Field(default_factory=dict)

class HistoryConfig(BaseModel):
    message_size: int = Field(default=15)
    load_message_size: int = Field(default=5)
    max_historical_days: int = Field(default=30)

class ChatConfig(BaseModel):
    system_prompt: Optional[str] = Field(default=None) # Deprecated if using prompts/ but kept for compatibility
    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)

class CoreAppConfig(BaseModel):
    """Configuration for the Core App (UI/Agent) instance."""
    company_name: str = Field(default="InvestorIQ")
    industry: str = Field(default="Investment")
    language: str = Field(default="en")
    system_language: str = Field(default="zh")
    max_concurrency: int = Field(default=1)
    
    users: List[UserConfig] = Field(default_factory=list)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    templates: Dict[str, str] = Field(default_factory=dict) # Legacy
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)

class QdrantConfig(BaseModel):
    """Qdrant vector database connection configuration."""
    url: Optional[str] = Field(default=None)  # Server endpoint (e.g., http://qdrant:6333)
    path: Optional[str] = Field(default=None)  # Local file storage (e.g., data/vectordb)
    location: Optional[str] = Field(default=None)  # In-memory storage (":memory:")
    api_key: Optional[str] = Field(default=None)  # Optional API authentication

class ExtractionConfig(BaseModel):
    ocr_model: str
    transcribe_model: str
    ocr_dpi: int
    pdf_process_method: str
    language: str
    ocr_stream: bool = Field(default=False)

class AnalyticsPipelineConfig(BaseModel):
    """Shared settings for the field_extraction → project_score pipeline."""
    db_path: str = Field(default="data/scorecard.db")
    write_files: bool = Field(default=False)

# Legacy aliases kept so that any remaining external code reading cfg.scorecard or
# cfg.analytics keeps working until those callers are updated.
ScorecardConfig = AnalyticsPipelineConfig
AnalyticsConfig = AnalyticsPipelineConfig

class VectorDBConfig(BaseModel):    
    qdrant: Optional[QdrantConfig] = Field(default_factory=QdrantConfig)  # Qdrant connection config
    embedding_model: str = Field(default="BAAI/bge-m3")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    # Legacy flat fields for backward compatibility (deprecated)
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_path: Optional[str] = Field(default=None)
    max_chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    transcript_window_seconds: int = Field(default=60)
    transcript_overlap_seconds: int = Field(default=15)
    llm: Optional[LLMConfig] = Field(default=None)

class ReportingConfig(BaseModel):
    default_template: str = Field(default="original_lite.json")
    generate_pdf: bool = Field(default=True)
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="qwen3:14b", context_window=65536))

class PodcastConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="qwen3:14b", temperature=0.7, context_window=65536))

class AppConfig(BaseModel):    
    api_key: str = Field(default="dev-secret-key")
    jwt_secret: str = Field(default="investor-iq-secret-change-me")
    project_dir: str = Field(default="projects")
    low_memory: bool = Field(default=True)
    
    # New core_app structure
    core_app: CoreAppConfig = Field(default_factory=CoreAppConfig)
    
    # General LLM setting used across the application
    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    
    # Legacy sections (partially migrated to core_app)
    # Keeping these for backward compatibility of other modules if needed, 
    # but core_app will use the one inside CoreAppConfig.
    chat: ChatConfig = Field(default_factory=ChatConfig)
    tenants: List[UserConfig] = Field(default_factory=list, alias="users_legacy")
    
    extraction: Optional[ExtractionConfig] = Field(default=None)
    analytics_pipeline: AnalyticsPipelineConfig = Field(default_factory=AnalyticsPipelineConfig)
    # Legacy keys — populated from config.yaml if present, for backward compatibility
    scorecard: Optional[AnalyticsPipelineConfig] = Field(default=None)
    analytics: Optional[AnalyticsPipelineConfig] = Field(default=None)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    podcast: PodcastConfig = Field(default_factory=PodcastConfig)

def load_config_from_path(config_path: Path) -> AppConfig:
    """Loads application configuration from a specific YAML file."""
    logging.info(f"Loading config from: {config_path}")
    if not config_path.exists():
        print(f"[config] WARNING: config file not found: {config_path}", file=sys.stderr)
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = AppConfig(**(data or {}))
        print(f"[config] Loaded: {config_path}", file=sys.stderr)
        return cfg
    except Exception as e:
        print(f"[config] FATAL: failed to parse {config_path}: {e}", file=sys.stderr)
        raise

def find_config() -> Path:
    """
    Determines which config file to use based on environment.
    Priority:
    1. PACKAGE_CONFIG_PATH env var
    2. packages/{PACKAGE_NAME}/config.yaml env var
    3. src/{PACKAGE_NAME}/config.yaml env var
    4. Root config.yaml
    """
    env_path = os.getenv("PACKAGE_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        _config_logger.warning("PACKAGE_CONFIG_PATH=%s does not exist — falling back", env_path)

    pkg_name = os.getenv("PACKAGE_NAME") or os.getenv("INVESTOR_IQ_PACKAGE")
    if pkg_name:
        pkg_cfg = Path("packages") / pkg_name / "config.yaml"
        if pkg_cfg.exists():
            return pkg_cfg

        src_cfg = Path("src") / pkg_name / "config.yaml"
        if src_cfg.exists():
            return src_cfg

        _config_logger.warning("PACKAGE_NAME=%s set but no config.yaml found — falling back to root", pkg_name)

    return CONFIG_PATH

def load_config() -> AppConfig:
    """Loads the appropriate configuration for the current context."""
    return load_config_from_path(find_config())

def reload_config():
    """Reloads the global configuration instance."""
    global config
    config = load_config()

# Global configuration instance
config = load_config()
