"""LLM provider utilities for initializing and managing language models.

Provides:
- LLM factory function (get_llm) supporting Ollama, OpenAI, DeepSeek, LlamaCpp
- Message content extraction with CoT/tool_call artifact removal
- Token usage extraction and estimation utilities
- Stop token configuration for different model families
"""

import os
import re
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import BaseMessage
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from pydantic import SecretStr

from src.common.logger import get_logger
from src.common.LLMProvider import LLMSettings, LLAMA_CPP_MODEL_PATHS
from src.common.ml_utils import get_device

load_dotenv()

logger = get_logger("llm_utils")

# Type alias for LLM chat models
LLMType = ChatDeepSeek | ChatOllama | ChatOpenAI | ChatLlamaCpp

_DEFAULT_LLM_CONTEXT_WINDOW = 16384

# Global cache for LLM instances to support lazy loading and reuse
_LLM_CACHE: dict[tuple, LLMType] = {}

def get_message_content(msg: BaseMessage) -> str:
    # 1. Get the raw content
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        content = " ".join([str(c) for c in msg.content])
    else:
        content = str(msg.content)
    
    # 2. Remove <think>...</think> blocks (Standard CoT)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # 3. Remove <tool_call>...</tool_call> blocks (Qwen Quirk)
    # WARNING: Only do this if the tool_call contains CHINESE/TEXT reasoning. 
    # If you actually use tools elsewhere, be careful. 
    # Based on your log, it contains "首先，分析...", so we must strip it.
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)

    # 4. Remove orphaned closing tags (common artifact)
    content = content.replace('</think>', '')
    content = content.replace('</tool_call>', '')

    return content


def get_token_count(messages: list[BaseMessage]) -> int:
    """Estimate token count for a list of messages.
        args:
            messages: List of BaseMessage objects.
        returns:
            Estimated token count."""
    total_tokens = 0
    for msg in messages:
        content = get_message_content(msg)
        # Conservative token estimation: 1 token per character (better for CJK)
        total_tokens += len(content)
    return total_tokens


def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def clear_gpu_memory():
    """Clear CUDA cache and run garbage collection to free up VRAM."""
    import gc
    try:
        if get_device() == 'cuda':
            import torch
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def unload_ollama_model(model_name: str):
    """Unload a specific model from Ollama memory to free up VRAM."""
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    try:
        logger.info(f"Requesting Ollama to unload model: {model_name}")
        # Sending a generate request with keep_alive=0 unloads the model immediately
        requests.post(f"{base_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=5)
    except Exception as e:
        logger.warning(f"Failed to unload Ollama model {model_name}: {e}")


def get_llm(llm_setting: LLMSettings) -> Any:
    """Initialize and return a UnifiedChatModel.
    
    Args:
        llm_setting: LLMSettings object
        
    Returns:
        Initialized UnifiedChatModel instance
    """
    from src.common.unified_llm import UnifiedChatModel
    
    return UnifiedChatModel(
        provider=llm_setting.provider,
        model_name=llm_setting.model_name,
        temperature=llm_setting.temperature,
        context_window=llm_setting.context_window,
        top_p=llm_setting.top_p,
        thinking=not llm_setting.no_thinking
    )


def get_llm_legacy(llm_setting: LLMSettings) -> LLMType:
    """Initialize and return a provider-specific LLM chat model (Legacy).
    
    Args:
        llm_setting: LLMSettings object
        
    Returns:
        Initialized chat model instance
        
    Raises:
        ValueError: If provider is unsupported or API key is missing
    """
    provider = llm_setting.provider
    model_name = llm_setting.model_name
    temperature = llm_setting.temperature
    context_window = llm_setting.context_window
    top_p = llm_setting.top_p
    no_thinking = getattr(llm_setting, 'no_thinking', False)

    # Check cache first to avoid re-initializing identical models
    cache_key = (provider, model_name, temperature, context_window, top_p, no_thinking)
    if cache_key in _LLM_CACHE:
        logger.debug(f"Using cached LLM provider: {provider} with model: {model_name}")
        return _LLM_CACHE[cache_key]

    logger.info(f"Initializing NEW LLM provider: {provider} with model: {model_name}")
    
    llm = None
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(f"DEEPSEEK_API_KEY environment variable not set")
        if no_thinking and "reasoner" in model_name:
            logger.info(f"no_thinking=True: Swapping deepseek-reasoner for deepseek-chat")
            model_name = "deepseek-chat"
            
        llm = ChatDeepSeek(
            model=model_name,
            api_key=SecretStr(api_key), # type: ignore
            temperature=0.0 if temperature is None else temperature,
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY environment variable not set")
        kwargs = {
            "model": model_name,
            "api_key": SecretStr(api_key), # type: ignore
            "temperature": 0.0 if temperature is None else temperature,
        }
        if no_thinking:
            kwargs["reasoning_effort"] = "low"
        llm = ChatOpenAI(**kwargs)
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        kwargs = {
            "model": model_name,
            "temperature": 0.0 if temperature is None else temperature,
            "top_p": top_p,
            "num_ctx": context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
            "base_url": base_url,
        }
        if no_thinking:
            # Langchain's ChatOllama supports a reasoning toggle, False strips <think> token generation natively
            kwargs["reasoning"] = False
            
        llm = ChatOllama(**kwargs)
    elif provider == "llama_cpp":
        model_path = LLAMA_CPP_MODEL_PATHS.get(llm_setting.model_name)
        if not model_path:
            raise ValueError(f"Model path not defined for LlamaCpp model: {llm_setting.model_name}")
        
        if no_thinking:
            logger.warning(f"no_thinking=True is not cleanly supported via API config on un-managed llama_cpp models. Falling back to content stripping.")
            
        stop_token = get_tokens_for_model(model_name)
        
        llm = ChatLlamaCpp(
            model_path=model_path,
            temperature=0.0 if temperature is None else temperature,
            n_gpu_layers=-1,  # Critical for M1 Pro speed
            max_tokens=8192,
            n_ctx=context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
            stop=stop_token,
            verbose=False  # <--- get rid of the wall of text (specifically for llama_cpp)
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if llm:
        # Attach settings for proxy serialization if possible
        try:
            # LangChain models are often Pydantic models with strict field validation.
            # We try to attach it anyway for the MediaVectorDB proxy to find.
            object.__setattr__(llm, "llm_setting", llm_setting)
        except Exception:
            # If we can't attach it, the proxy will just use defaults or the server's config.
            pass
        
        # Store in cache before returning
        _LLM_CACHE[cache_key] = llm
        
    return llm


def extract_token_usage(response) -> dict:
    """Extract token usage information from LLM response.
    
    Handles different response types and safely extracts usage metadata.
    
    Args:
        response: LLM response object (AIMessage, ChatResponse, etc.)
        
    Returns:
        dict: Token usage with keys 'input_tokens', 'output_tokens', 'total_tokens'
              Returns dict with zero values if no usage info available
    """
    usage = {}
    
    # Try to get usage_metadata from response (LangChain AIMessage)
    if hasattr(response, 'usage_metadata') and isinstance(response.usage_metadata, dict):
        usage = response.usage_metadata.copy()
    
    # Try to get from response_metadata (some LLM responses)
    elif hasattr(response, 'response_metadata') and isinstance(response.response_metadata, dict):
        if 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage'].copy()
    
    # Ensure standard keys exist
    if not usage:
        usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    return usage


def get_ollama_context_length(model: str, base_url: str = "http://localhost:11434") -> int | None:
    r = requests.post(f"{base_url}/api/show", json={"name": model})
    r.raise_for_status()
    info = r.json()
    logger.debug(f"Ollama model info: {info}")
    # Many LLaMA-style models expose this key:
    ctx = info.get("model_info", {}).get("llama.context_length")
    return ctx  # e.g. 8192


STOP_TOKENS = {
    # --- Qwen Family (Includes DeepSeek-R1-Distill-Qwen) ---
    # Standard ChatML format
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    
    # --- Llama 3 Family (Includes DeepSeek-R1-Distill-Llama) ---
    # <|eot_id|> ends the turn (Assistant finished)
    # <|end_of_text|> ends the generation completely
    "llama3": ["<|eot_id|>", "<|end_of_text|>"],
    
    # --- Mistral / Mixtral Family ---
    # Classic EOS token. (Newer Mistral Large may differ, but this is standard)
    "mistral": ["</s>"],
    
    # --- Gemma 2 Family ---
    "gemma": ["<end_of_turn>", "<eos>"],
    
    # --- Phi 3 / 3.5 Family ---
    "phi3": ["<|end|>", "<|endoftext|>"],
    
    # --- DeepSeek V2/V3 (Base/Raw) ---
    # Only if running original DeepSeek architecture, not distillations
    "deepseek_base": ["<|EOT|>", "<|end_of_sentence|>"],
    
    # --- Fallback/Generic ---
    "default": []
}

def get_tokens_for_model(model_name_or_path: str) -> list:
    """
    Returns the correct stop tokens by analyzing the model name string.
    """
    s = model_name_or_path.lower()
    
    if "qwen" in s:
        return STOP_TOKENS["qwen"]
    elif "llama-3" in s or "llama3" in s:
        return STOP_TOKENS["llama3"]
    elif "mistral" in s or "mixtral" in s:
        return STOP_TOKENS["mistral"]
    elif "gemma" in s:
        return STOP_TOKENS["gemma"]
    elif "phi-3" in s or "phi3" in s:
        return STOP_TOKENS["phi3"]
    
    # DeepSeek Handling: Check if it's a Qwen/Llama distillation or Base
    elif "deepseek" in s:
        if "llama" in s: return STOP_TOKENS["llama3"] # Distill-Llama
        if "qwen" in s:  return STOP_TOKENS["qwen"]   # Distill-Qwen
        return STOP_TOKENS["deepseek_base"]           # Raw DeepSeek
        
    return STOP_TOKENS["default"]


def detect_output_language(text: str) -> str:
    if not text:
        return "same as user's request"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "Chinese"
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "Japanese"
    if any("\uac00" <= ch <= "\ud7af" for ch in text):
        return "Korean"
    if any("\u0400" <= ch <= "\u04ff" for ch in text):
        return "Russian/Cyrillic"
    if any("\u0600" <= ch <= "\u06ff" for ch in text):
        return "Arabic"
    return "English"