import os
import sys
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from src.common.unified_llm import UnifiedChatModel
from src.common.logger import get_logger

logger = get_logger("integration_test")

class Investor(BaseModel):
    """Information about an investor."""
    name: str = Field(description="Name of the investor")
    focus_areas: list[str] = Field(description="Industries they invest in")
    min_check_size: float = Field(description="Minimum investment amount in USD")

def test_provider(provider: str, model_name: str):
    logger.info(f"--- Testing {provider}/{model_name} ---")

    try:
        llm = UnifiedChatModel(provider=provider, model_name=model_name, temperature=0.0)

        # 1. Basic Invoke (String)
        logger.info("Test 1: Basic Invoke (String)")
        resp = llm.invoke("Say 'Hello' and nothing else.")
        logger.info(f"Response: {resp.content}")

        # 2. Invoke with Message List
        logger.info("Test 2: Invoke with Message List")
        messages = [
            SystemMessage(content="You are a minimalist assistant."),
            HumanMessage(content="What is 2+2? Answer with one word.")
        ]
        resp_msg = llm.invoke(messages)
        logger.info(f"Response: {resp_msg.content}")

        # 3. Streaming
        logger.info("Test 3: Streaming Output")
        print("Streaming: ", end="", flush=True)
        stream_resp = ""
        for chunk in llm.stream("Count from 1 to 5 slowly with words."):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(content, end="", flush=True)
            stream_resp += content
        print("\nStream Finished.")

        # 4. Structured Output (Invoke with Schema)
        logger.info("Test 4: Structured Output (Invoke with Schema)")
        prompt = "Extract info: Bob is a VC focusing on Web3 and Gaming. He invests at least 1,000,000 USD."
        investor = llm.invoke(prompt, schema=Investor)
        logger.info(f"Parsed Object: {investor}")

        # 5. Tool Binding & Invoke
        logger.info("Test 5: Tool Binding & Invoke")
        def get_current_time():
            """Get the current time."""
            return "12:00 PM"

        llm_with_tools = llm.bind_tools([get_current_time])
        resp_tool = llm_with_tools.invoke("What time is it?")
        # Check if it tried to call the tool (provider dependent)
        has_tool_call = hasattr(resp_tool, "additional_kwargs") and "tool_calls" in resp_tool.additional_kwargs
        logger.info(f"Tool Call Detected: {has_tool_call}")

    except Exception as e:
        logger.error(f"Error testing {provider}: {e}", exc_info=True)

if __name__ == "__main__":
    # Test Ollama
    test_provider("ollama", "qwen3:14b")

    # Test DeepSeek (if API key available)
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\n" + "="*50 + "\n")
        test_provider("deepseek", "deepseek-chat")
    else:
        logger.warning("Skipping DeepSeek: DEEPSEEK_API_KEY not found.")
