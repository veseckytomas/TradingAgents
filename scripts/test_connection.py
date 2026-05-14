import os
import httpx
from langchain_openai import ChatOpenAI
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

# Configuration from the user's setup
os.environ["DEEPSEEK_API_KEY"] = "sk-90a59f72ca0b4643bb9396466af5f684"
base_url = "https://llm.ai.e-infra.cz/v1"
model = "deepseek-v4-pro-thinking"

print(f"Testing connection to {base_url} with model {model}...")

try:
    # Attempt 1: Standard client (strict SSL)
    print("\nAttempt 1: Strict SSL (Default)...")
    llm = DeepSeekChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=10,
        max_retries=0
    )
    response = llm.invoke("hi")
    print("Success with strict SSL!")
    print(f"Response: {response.content}")

except Exception as e:
    print(f"Failed with strict SSL: {type(e).__name__}: {e}")
    
    try:
        # Attempt 2: Ignore SSL
        print("\nAttempt 2: Ignoring SSL errors...")
        http_client = httpx.Client(verify=False)
        llm_no_ssl = DeepSeekChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            http_client=http_client,
            timeout=10,
            max_retries=0
        )
        response = llm_no_ssl.invoke("hi")
        print("Success when ignoring SSL!")
        print(f"Response: {response.content}")
        print("\nDIAGNOSIS: The issue is indeed SSL verification.")
    except Exception as e2:
        print(f"Failed even when ignoring SSL: {type(e2).__name__}: {e2}")
