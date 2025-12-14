"""
Combined FastAPI + Streamlit deployment for Railway
"""
import os
import subprocess
import threading
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import your FastAPI app
import sys
sys.path.append('backend')
from backend.main import app as fastapi_app

def run_streamlit():
    """Run Streamlit in a separate thread"""
    time.sleep(5)  # Wait for FastAPI to start
    os.environ["API_BASE_URL"] = "http://localhost:8000"
    subprocess.run([
        "streamlit", "run", "frontend/app.py", 
        "--server.port=8501", 
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ])

if __name__ == "__main__":
    # Start Streamlit in background
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Start FastAPI
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)