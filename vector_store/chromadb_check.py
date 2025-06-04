import os
import urllib.parse
import requests
from dotenv import load_dotenv

load_dotenv()

raw_host = os.getenv("CHROMA_URL_EXTERNAL", "").strip()
basic_auth = os.getenv("BASIC_AUTH_HEADER", "").strip()

parsed = urllib.parse.urlparse(raw_host if raw_host.startswith("http") else "http://" + raw_host)
HOST = parsed.hostname  
PORT = 443              

BASE_URL = f"https://{HOST}:{PORT}"
HEADERS = {
    "Authorization": basic_auth,
    "Content-Type": "application/json"
}

if __name__ == "__main__":
    try:
        resp = requests.get(f"{BASE_URL}/api/v2/heartbeat", headers=HEADERS)
        resp.raise_for_status()
        print("✅ heartbeat 응답:", resp.json())  
    except Exception as e:
        print("❌ heartbeat 호출 실패:", e, getattr(e, "response", None))
