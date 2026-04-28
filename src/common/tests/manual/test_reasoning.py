import os
from langchain_core.messages import HumanMessage
from src.common.unified_llm import UnifiedChatModel
from src.common.logger import get_logger

logger = get_logger("reasoning_test")

def test_reasoning(provider: str, model_name: str):
    logger.info(f"--- Testing Reasoning Toggles for {provider}/{model_name} ---")

    question = "If I have 3 apples and eat 1, how many do I have? Explain your reasoning briefly."

    # 1. Test with Thinking ENABLED (thinking=True)
    logger.info("Case A: Reasoning ENABLED (thinking=True)")
    llm_with_thinking = UnifiedChatModel(
        provider=provider,
        model_name=model_name,
        thinking=True,
        temperature=0.0
    )
    resp_a = llm_with_thinking.invoke(question)
    logger.info(f"Response (With Thinking):\n{resp_a.content}")
    # Note: Some models return <think> tags in content, others strip them to a separate field.
    # UnifiedChatModel currently proxies the raw response.

    # 2. Test with Thinking DISABLED (thinking=False)
    logger.info("Case B: Reasoning DISABLED (thinking=False)")
    llm_no_thinking = UnifiedChatModel(
        provider=provider,
        model_name=model_name,
        thinking=False,
        temperature=0.0
    )
    resp_b = llm_no_thinking.invoke(question)
    logger.info(f"Response (No Thinking):\n{resp_b.content}")

if __name__ == "__main__":
    # Test Ollama with Qwen (often used for distilled R1 models)
    test_reasoning("ollama", "qwen3:14b")

    # Test DeepSeek (if API key available)
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\n" + "="*50 + "\n")
        # For DeepSeek, 'deepseek-reasoner' is the model that supports thinking
        # Our implementation swaps it to 'deepseek-chat' if thinking=False
        test_reasoning("deepseek", "deepseek-reasoner")
    else:
        logger.warning("Skipping DeepSeek: DEEPSEEK_API_KEY not found.")
