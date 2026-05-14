import os
import httpx
from langchain_openai import ChatOpenAI
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

# Configuration from the user's setup
os.environ["DEEPSEEK_API_KEY"] = "sk-90a59f72ca0b4643bb9396466af5f684"
base_url = "https://llm.ai.e-infra.cz/v1"
model = "deepseek-v4-pro-thinking"

print(f"Testing connection to {base_url} with model {model}...")

# Simulation of TradingAgentsGraph's logic
llm_kwargs = {
    "model": model,
    "base_url": base_url,
    "api_key": os.environ["DEEPSEEK_API_KEY"],
}

# The CLI also might pass use_responses_api if provider is openai, 
# but for deepseek it should use standard chat completions.

try:
    print("\nAttempt 1: DeepSeekChatOpenAI instantiation (matching provider='deepseek')...")
    llm = DeepSeekChatOpenAI(**llm_kwargs)
    response = llm.invoke("hi")
    print("Success with DeepSeekChatOpenAI!")
    print(f"Response: {response.content}")

except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
