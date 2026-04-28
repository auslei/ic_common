import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration

from src.common.unified_llm import UnifiedChatModel, clean_reasoning_content, _RAW_LLM_CACHE


class SampleSchema(BaseModel):
    """Sample schema for testing structured output."""
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")


class TestCleanReasoningContent:

    def test_strips_think_block(self):
        content = "<think>internal reasoning</think>Final answer"
        assert clean_reasoning_content(content) == "Final answer"

    def test_strips_tool_call_block(self):
        content = "<tool_call>call_data</tool_call>Result"
        assert clean_reasoning_content(content) == "Result"

    def test_strips_orphaned_closing_tags(self):
        content = "Some text</think>more</tool_call>"
        result = clean_reasoning_content(content)
        assert "</think>" not in result
        assert "</tool_call>" not in result

    def test_returns_non_string_unchanged(self):
        assert clean_reasoning_content(42) == 42
        assert clean_reasoning_content(None) is None

    def test_adjacent_blocks(self):
        content = "<think>reasoning</think><tool_call>call</tool_call>answer"
        assert clean_reasoning_content(content) == "answer"

    def test_multiline_think_block(self):
        content = "<think>\nstep 1\nstep 2\n</think>conclusion"
        assert clean_reasoning_content(content) == "conclusion"


class TestUnifiedChatModel:

    @pytest.fixture
    def mock_llm(self):
        """Creates a mock LangChain LLM instance."""
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Mocked response")
        llm.bind_tools.return_value = MagicMock()
        llm.bind.return_value = MagicMock()
        llm.stream.return_value = iter([AIMessageChunk(content="M"), AIMessageChunk(content="o")])
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Async mocked response"))
        return llm

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_initialization(self, mock_init, mock_llm):
        """Test that initialization correctly sets parameters and calls _init_underlying_model."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(
            provider="ollama",
            model_name="qwen3:14b",
            temperature=0.7,
            context_window=4096,
            thinking=False
        )

        assert model.provider == "ollama"
        assert model.model_name == "qwen3:14b"
        assert model.temperature == 0.7
        assert model.context_window == 4096
        assert model.thinking is False

        # Verify _init_underlying_model was called with correct parameters
        mock_init.assert_called_once_with(
            provider="ollama",
            model_name="qwen3:14b",
            temperature=0.7,
            context_window=4096,
            top_p=1.0,
            thinking=False
        )

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_basic_invoke(self, mock_init, mock_llm):
        """Test simple string invocation."""
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        response = model.invoke("Hello")

        assert response.content == "Mocked response"
        mock_llm.invoke.assert_called_once_with("Hello", config=None)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_invoke_string(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test invoke with a Pydantic schema using a string input."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Prompt with schema"
        mock_parse.return_value = SampleSchema(name="Alice", age=30)

        model = UnifiedChatModel(provider="deepseek", model_name="deepseek-chat")

        result = model.invoke("Who is Alice?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Alice"
        assert result.age == 30

        mock_build.assert_called_once_with("Who is Alice?", SampleSchema)
        mock_llm.invoke.assert_called_once_with("Prompt with schema", config=None)
        mock_parse.assert_called_once_with("Mocked response", SampleSchema, llm=mock_llm)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_invoke_list(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test invoke with a Pydantic schema using a list of messages."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Modified message"
        mock_parse.return_value = SampleSchema(name="Bob", age=40)

        model = UnifiedChatModel(provider="ollama", model_name="llama3")
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about Bob.")
        ]

        result = model.invoke(messages, schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Bob"

        # Verify that only the last human message was modified
        args, kwargs = mock_llm.invoke.call_args
        sent_messages = args[0]
        assert len(sent_messages) == 2
        assert sent_messages[0].content == "You are a helpful assistant."
        assert sent_messages[1].content == "Modified message"
        mock_build.assert_called_once_with("Tell me about Bob.", SampleSchema)
        mock_parse.assert_called_once_with("Mocked response", SampleSchema, llm=mock_llm)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_bind_tools(self, mock_init, mock_llm):
        """Test that bind_tools returns a new UnifiedChatModel wrapping the bound model."""
        mock_init.return_value = mock_llm
        bound_mock = MagicMock()
        mock_llm.bind_tools.return_value = bound_mock

        model = UnifiedChatModel(provider="openai", model_name="gpt-4", temperature=0.5)
        tools = [lambda x: x]

        new_model = model.bind_tools(tools)

        assert isinstance(new_model, UnifiedChatModel)
        assert new_model.provider == "openai"
        assert new_model.temperature == 0.5
        assert new_model.model == bound_mock
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_bind_parameters(self, mock_init, mock_llm):
        """Test that bind returns a new UnifiedChatModel wrapping the bound model."""
        mock_init.return_value = mock_llm
        bound_mock = MagicMock()
        mock_llm.bind.return_value = bound_mock

        model = UnifiedChatModel(provider="ollama", model_name="qwen2.5")

        new_model = model.bind(stop=["\n"])

        assert isinstance(new_model, UnifiedChatModel)
        assert new_model.model == bound_mock
        mock_llm.bind.assert_called_once_with(stop=["\n"])

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_stream_proxy(self, mock_init, mock_llm):
        """Test that stream is correctly proxied."""
        mock_init.return_value = mock_llm
        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        list(model.stream("Hello"))  # consume iterator

        mock_llm.stream.assert_called_once_with("Hello", config=None)

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_invoke_thinking_override_creates_variant(self, mock_init, mock_llm):
        """Test that passing thinking= at call time creates a variant model."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=True)

        # Call with thinking=False — should create a variant via a second init call
        model.invoke("Hello", thinking=False)

        # _init_underlying_model called once for base model + once for variant
        assert mock_init.call_count == 2
        # Second call should have thinking=False
        _, variant_kwargs = mock_init.call_args
        assert variant_kwargs["thinking"] is False

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    def test_invoke_no_thinking_cleans_content(self, mock_init, mock_llm):
        """Test that thinking=False strips <think> tags from the response."""
        mock_llm.invoke.return_value = AIMessage(content="<think>reasoning</think>Clean answer")
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=False)
        response = model.invoke("Hello")

        assert response.content == "Clean answer"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_ainvoke_basic(self, mock_init, mock_llm):
        """Test basic async invocation."""
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        response = await model.ainvoke("Hello")

        assert response.content == "Async mocked response"
        mock_llm.ainvoke.assert_called_once_with("Hello", config=None)

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.build_structured_prompt")
    @patch("src.common.unified_llm.parse_structured_output")
    async def test_ainvoke_structured_output(self, mock_parse, mock_build, mock_init, mock_llm):
        """Test async invoke with Pydantic schema."""
        mock_init.return_value = mock_llm
        mock_build.return_value = "Prompt with schema"
        mock_parse.return_value = SampleSchema(name="Carol", age=25)

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        result = await model.ainvoke("Who is Carol?", schema=SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.name == "Carol"
        assert result.age == 25
        mock_build.assert_called_once_with("Who is Carol?", SampleSchema)

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_ainvoke_no_thinking_cleans_content(self, mock_init, mock_llm):
        """Test that async invoke with thinking=False strips <think> tags."""
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="<think>step</think>Answer"))
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="ollama", model_name="qwen3:14b", thinking=False)
        response = await model.ainvoke("Hello")

        assert response.content == "Answer"

    @pytest.mark.asyncio
    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    async def test_astream_proxy(self, mock_init, mock_llm):
        """Test that astream is correctly proxied."""
        async def mock_astream(*args, **kwargs):
            yield AIMessageChunk(content="A")
            yield AIMessageChunk(content="B")

        mock_llm.astream = mock_astream
        mock_init.return_value = mock_llm

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")
        chunks = [chunk async for chunk in model.astream("Hello")]

        assert len(chunks) == 2
        assert chunks[0].content == "A"
        assert chunks[1].content == "B"

    @patch("src.common.unified_llm._RAW_LLM_CACHE", {})
    @patch("src.common.unified_llm.ChatOllama")
    def test_cache_reuse(self, mock_ollama_cls):
        """Test that identical init params return the cached model instance."""
        mock_instance = MagicMock()
        mock_ollama_cls.return_value = mock_instance

        result1 = UnifiedChatModel._init_underlying_model(
            provider="ollama", model_name="llama3", temperature=0.0,
            context_window=4096, top_p=1.0, thinking=True
        )
        result2 = UnifiedChatModel._init_underlying_model(
            provider="ollama", model_name="llama3", temperature=0.0,
            context_window=4096, top_p=1.0, thinking=True
        )

        assert result1 is result2
        mock_ollama_cls.assert_called_once()  # Only constructed once

    def test_invalid_provider_raises(self):
        """Test that an unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            UnifiedChatModel._init_underlying_model(
                provider="unknown_provider", model_name="model",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    def test_missing_openai_api_key_raises(self, monkeypatch):
        """Test that missing OPENAI_API_KEY raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            UnifiedChatModel._init_underlying_model(
                provider="openai", model_name="gpt-4",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    def test_missing_deepseek_api_key_raises(self, monkeypatch):
        """Test that missing DEEPSEEK_API_KEY raises ValueError."""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            UnifiedChatModel._init_underlying_model(
                provider="deepseek", model_name="deepseek-chat",
                temperature=0.0, context_window=None, top_p=1.0, thinking=True
            )

    @patch("src.common.unified_llm.UnifiedChatModel._init_underlying_model")
    @patch("src.common.unified_llm.parse_structured_output")
    def test_structured_output_parse_failure_propagates(self, mock_parse, mock_init, mock_llm):
        """Test that a parse failure in invoke raises the exception."""
        mock_init.return_value = mock_llm
        mock_parse.side_effect = ValueError("Parse failed")

        model = UnifiedChatModel(provider="openai", model_name="gpt-4")

        with pytest.raises(ValueError, match="Parse failed"):
            model.invoke("Who is Alice?", schema=SampleSchema)
