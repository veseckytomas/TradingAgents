import os
import httpx
import yfinance as yf
from langchain_openai import ChatOpenAI
from tradingagents.llm_clients.openai_client import DeepSeekChatOpenAI

# Configuration from the user's setup
os.environ["DEEPSEEK_API_KEY"] = "sk-90a59f72ca0b4643bb9396466af5f684"
base_url = "https://llm.ai.e-infra.cz/v1"
model = "deepseek-v4-pro-thinking"

print(f"Testing connectivity components...")

# 1. Test LLM
try:
    print("\n--- Testing LLM ---")
    llm = DeepSeekChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        timeout=10
    )
    response = llm.invoke("hi")
    print("LLM: SUCCESS")
except Exception as e:
    print(f"LLM: FAILED: {type(e).__name__}: {e}")

# 2. Test yfinance (often the source of 'connection errors')
try:
    print("\n--- Testing yfinance ---")
    ticker = "SPY"
    print(f"Fetching history for {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        print("yfinance: SUCCESS")
    else:
        print("yfinance: SUCCESS (but empty data)")
except Exception as e:
    print(f"yfinance: FAILED: {type(e).__name__}: {e}")

# 3. Test generic requests (proxy check)
try:
    print("\n--- Testing generic HTTPS ---")
    resp = httpx.get("https://www.google.com", timeout=5)
    print(f"HTTPS: SUCCESS (Status {resp.status_code})")
except Exception as e:
    print(f"HTTPS: FAILED: {type(e).__name__}: {e}")
