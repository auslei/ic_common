import os
import re
import time
from typing import Any, List, Optional, Union, Dict, Type, Iterator
from pydantic import BaseModel, Field, ConfigDict, SecretStr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Provider-specific LangChain models
from langchain_community.chat_models import ChatLlamaCpp
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

from src.common.pydantic_helper import build_structured_prompt, parse_structured_output
from src.common.logger import get_logger

logger = get_logger("unified_llm", icon="🧠")

# Internal constants and cache for fully independent operation
_DEFAULT_LLM_CONTEXT_WINDOW = 16384
_RAW_LLM_CACHE: Dict[tuple, Any] = {}

# LlamaCpp model paths
_BASE_BLOBS_LOCATION = "/usr/share/ollama/.ollama/models/blobs"
LLAMA_CPP_MODEL_PATHS = {
    "qwen3:14b": f"{_BASE_BLOBS_LOCATION}/sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e",
    "qwen3:8b": f"{_BASE_BLOBS_LOCATION}/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
}

STOP_TOKENS = {
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    "llama3": ["<|eot_id|>", "<|end_of_text|>"],
    "mistral": ["</s>"],
    "gemma": ["<end_of_turn>", "<eos>"],
    "phi3": ["<|end|>", "<|endoftext|>"],
    "deepseek_base": ["<|EOT|>", "<|end_of_sentence|>"],
    "default": []
}

def get_tokens_for_model(model_name_or_path: str) -> list:
    """Helper to determine stop tokens based on model family."""
    s = model_name_or_path.lower()
    if "qwen" in s: return STOP_TOKENS["qwen"]
    if "llama-3" in s or "llama3" in s: return STOP_TOKENS["llama3"]
    if "mistral" in s or "mixtral" in s: return STOP_TOKENS["mistral"]
    if "gemma" in s: return STOP_TOKENS["gemma"]
    if "phi-3" in s or "phi3" in s: return STOP_TOKENS["phi3"]
    if "deepseek" in s:
        if "llama" in s: return STOP_TOKENS["llama3"]
        if "qwen" in s:  return STOP_TOKENS["qwen"]
        return STOP_TOKENS["deepseek_base"]
    return STOP_TOKENS["default"]

def clean_reasoning_content(content: str) -> str:
    """Remove <think> and <tool_call> tags from content."""
    if not isinstance(content, str):
        return content
    # Remove <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # Remove <tool_call>...</tool_call> blocks
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
    # Remove orphaned closing tags
    content = content.replace('</think>', '')
    content = content.replace('</tool_call>', '')
    return content.strip()

class UnifiedChatModel(BaseChatModel):
    """
    A unified, independent wrapper for LangChain chat models that supports 
    provider-specific initialization, parameter retrieval, and automatic 
    structured output parsing.
    """
    provider: str
    model_name: str
    temperature: float = 0.0
    context_window: Optional[int] = None
    top_p: Optional[float] = 1.0
    thinking: bool = True
    
    # Internal model instance excluded from standard Pydantic validation/serialization
    model: Any = Field(exclude=True, default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def model_post_init(self, __context: Any) -> None:
        """Automatically initialize the underlying model after Pydantic validation."""
        if self.model is None:
            start_time = time.perf_counter()
            self.model = self._init_underlying_model(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                context_window=self.context_window,
                top_p=self.top_p,
                thinking=self.thinking
            )
            load_time = time.perf_counter() - start_time
            logger.debug(f"Underlying LLM {self.provider}/{self.model_name} ready in {load_time:.3f}s")

    @staticmethod
    def _init_underlying_model(
        provider: str, 
        model_name: str, 
        temperature: float, 
        context_window: Optional[int], 
        top_p: Optional[float], 
        thinking: bool
    ) -> Any:
        """Self-contained logic to initialize the provider-specific model."""
        
        # Check cache to avoid re-initializing identical models
        cache_key = (provider, model_name, temperature, context_window, top_p, thinking)
        if cache_key in _RAW_LLM_CACHE:
            return _RAW_LLM_CACHE[cache_key]

        logger.info(f"Initializing NEW raw LLM: {provider}/{model_name} (thinking={thinking})")
        
        llm = None
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            
            effective_model = model_name
            if not thinking and "reasoner" in model_name:
                logger.info("thinking=False: Swapping deepseek-reasoner for deepseek-chat")
                effective_model = "deepseek-chat"
                
            llm = ChatDeepSeek(
                model=effective_model,
                api_key=SecretStr(api_key),
                temperature=0.0 if temperature is None else temperature,
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            kwargs = {
                "model": model_name,
                "api_key": SecretStr(api_key),
                "temperature": 0.0 if temperature is None else temperature,
            }
            if not thinking:
                kwargs["reasoning_effort"] = "low"
            llm = ChatOpenAI(**kwargs)
        elif provider == "lms":
            base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            llm = ChatOpenAI(
                model=model_name,
                api_key=SecretStr("lm-studio"),  # LM Studio ignores the key
                base_url=base_url,
                temperature=0.0 if temperature is None else temperature,
            )
        elif provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
            kwargs = {
                "model": model_name,
                "temperature": 0.0 if temperature is None else temperature,
                "top_p": top_p,
                "num_ctx": context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
                "base_url": base_url,
            }
            if not thinking:
                kwargs["reasoning"] = False
            llm = ChatOllama(**kwargs)
        elif provider == "llama_cpp":
            model_path = LLAMA_CPP_MODEL_PATHS.get(model_name)
            if not model_path:
                raise ValueError(f"Model path not defined for LlamaCpp model: {model_name}")
            
            stop_token = get_tokens_for_model(model_name)
            llm = ChatLlamaCpp(
                model_path=model_path,
                temperature=0.0 if temperature is None else temperature,
                n_gpu_layers=-1,
                max_tokens=8192,
                n_ctx=context_window if context_window is not None else _DEFAULT_LLM_CONTEXT_WINDOW,
                stop=stop_token,
                verbose=False
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        _RAW_LLM_CACHE[cache_key] = llm
        return llm

    @property
    def _llm_type(self) -> str:
        return f"unified_{self.provider}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Internal LangChain entry point for generating completions."""
        start_time = time.perf_counter()
        if hasattr(self.model, "_generate"):
            result = self.model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        else:
            response = self.model.invoke(messages, stop=stop, **kwargs)
            result = ChatResult(generations=[ChatGeneration(message=response)])
        
        duration = time.perf_counter() - start_time
        logger.debug(f"Inference completed in {duration:.3f}s")
            
        # Clean reasoning if thinking is disabled
        if not self.thinking:
            for gen in result.generations:
                if hasattr(gen.message, "content") and isinstance(gen.message.content, str):
                    gen.message.content = clean_reasoning_content(gen.message.content)
                    
        return result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Internal LangChain entry point for async generating completions."""
        start_time = time.perf_counter()
        if hasattr(self.model, "_agenerate"):
            result = await self.model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        else:
            response = await self.model.ainvoke(messages, stop=stop, **kwargs)
            result = ChatResult(generations=[ChatGeneration(message=response)])
        
        duration = time.perf_counter() - start_time
        logger.debug(f"Async inference completed in {duration:.3f}s")
            
        # Clean reasoning if thinking is disabled
        if not self.thinking:
            for gen in result.generations:
                if hasattr(gen.message, "content") and isinstance(gen.message.content, str):
                    gen.message.content = clean_reasoning_content(gen.message.content)
                    
        return result

    def invoke(
        self, 
        input: Any, 
        config: Optional[Any] = None, 
        *, 
        schema: Optional[Type[BaseModel]] = None, 
        thinking: Optional[bool] = None, 
        **kwargs: Any
    ) -> Union[BaseMessage, Any]:
        """
        Invoke the model. Supports dynamic 'thinking' override at call time.
        """
        # Handle dynamic reasoning override
        if thinking is not None and thinking != self.thinking:
            variant = UnifiedChatModel(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                context_window=self.context_window,
                top_p=self.top_p,
                thinking=thinking
            )
            return variant.invoke(input, config=config, schema=schema, **kwargs)

        # Handle structured output
        if schema:
            logger.info(f"Invoking with structured output schema: {schema.__name__}")
            if isinstance(input, str):
                input = build_structured_prompt(input, schema)
            elif isinstance(input, list) and len(input) > 0:
                last_msg = input[-1]
                if isinstance(last_msg, (HumanMessage, SystemMessage)) and isinstance(last_msg.content, str):
                    input = list(input)
                    msg_cls = type(last_msg)
                    input[-1] = msg_cls(content=build_structured_prompt(last_msg.content, schema), **last_msg.additional_kwargs)

        start_time = time.perf_counter()
        response = self.model.invoke(input, config=config, **kwargs)
        duration = time.perf_counter() - start_time
        logger.info(f"Inference completed in {duration:.3f}s ({self.provider}/{self.model_name})")
        
        # Clean reasoning if thinking is disabled
        if not self.thinking and hasattr(response, "content") and isinstance(response.content, str):
            response.content = clean_reasoning_content(response.content)
        
        # Parse structured output
        if schema:
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            
            try:
                parsed = parse_structured_output(content, schema, llm=self.model)
                logger.debug(f"Successfully parsed structured output for {schema.__name__}")
                return parsed
            except Exception as e:
                logger.error(f"Failed to parse structured output: {e}")
                raise
        
        return response

    async def ainvoke(
        self, 
        input: Any, 
        config: Optional[Any] = None, 
        *, 
        schema: Optional[Type[BaseModel]] = None, 
        thinking: Optional[bool] = None, 
        **kwargs: Any
    ) -> Union[BaseMessage, Any]:
        """
        Async version of invoke.
        """
        # Handle dynamic reasoning override
        if thinking is not None and thinking != self.thinking:
            variant = UnifiedChatModel(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                context_window=self.context_window,
                top_p=self.top_p,
                thinking=thinking
            )
            return await variant.ainvoke(input, config=config, schema=schema, **kwargs)

        # Handle structured output
        if schema:
            logger.info(f"Invoking (async) with structured output schema: {schema.__name__}")
            if isinstance(input, str):
                input = build_structured_prompt(input, schema)
            elif isinstance(input, list) and len(input) > 0:
                last_msg = input[-1]
                if isinstance(last_msg, (HumanMessage, SystemMessage)) and isinstance(last_msg.content, str):
                    input = list(input)
                    msg_cls = type(last_msg)
                    input[-1] = msg_cls(content=build_structured_prompt(last_msg.content, schema), **last_msg.additional_kwargs)

        start_time = time.perf_counter()
        response = await self.model.ainvoke(input, config=config, **kwargs)
        duration = time.perf_counter() - start_time
        logger.info(f"Async inference completed in {duration:.3f}s ({self.provider}/{self.model_name})")
        
        # Clean reasoning if thinking is disabled
        if not self.thinking and hasattr(response, "content") and isinstance(response.content, str):
            response.content = clean_reasoning_content(response.content)
        
        # Parse structured output
        if schema:
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
            
            try:
                parsed = parse_structured_output(content, schema, llm=self.model)
                logger.debug(f"Successfully parsed structured output for {schema.__name__}")
                return parsed
            except Exception as e:
                logger.error(f"Failed to parse structured output: {e}")
                raise
        
        return response

    def stream(
        self, 
        input: Any, 
        config: Optional[Any] = None, 
        *, 
        thinking: Optional[bool] = None, 
        **kwargs: Any
    ) -> Iterator[BaseMessage]:
        """Proxy stream calls to the underlying model with dynamic reasoning override."""
        if thinking is not None and thinking != self.thinking:
            variant = UnifiedChatModel(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                context_window=self.context_window,
                top_p=self.top_p,
                thinking=thinking
            )
            yield from variant.stream(input, config=config, **kwargs)
            return

        yield from self.model.stream(input, config=config, **kwargs)

    async def astream(
        self, 
        input: Any, 
        config: Optional[Any] = None, 
        *, 
        thinking: Optional[bool] = None, 
        **kwargs: Any
    ):
        """Async proxy stream calls with dynamic reasoning override."""
        if thinking is not None and thinking != self.thinking:
            variant = UnifiedChatModel(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                context_window=self.context_window,
                top_p=self.top_p,
                thinking=thinking
            )
            async for chunk in variant.astream(input, config=config, **kwargs):
                yield chunk
            return

        async for chunk in self.model.astream(input, config=config, **kwargs):
            yield chunk

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "UnifiedChatModel":
        """
        Binds tools to the model and returns a new UnifiedChatModel instance.
        """
        logger.debug(f"Binding {len(tools)} tools to {self.provider}/{self.model_name}")
        bound_model = self.model.bind_tools(tools, **kwargs)
        return UnifiedChatModel(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            context_window=self.context_window,
            top_p=self.top_p,
            thinking=self.thinking,
            model=bound_model
        )

    def bind(self, **kwargs: Any) -> "UnifiedChatModel":
        """
        Binds parameters to the model and returns a new UnifiedChatModel instance.
        """
        logger.debug(f"Binding parameters {list(kwargs.keys())} to {self.provider}/{self.model_name}")
        bound_model = self.model.bind(**kwargs)
        return UnifiedChatModel(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            context_window=self.context_window,
            top_p=self.top_p,
            thinking=self.thinking,
            model=bound_model
        )
