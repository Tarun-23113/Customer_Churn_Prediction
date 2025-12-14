"""
Streamlit app entry point for Streamlit Cloud deployment
"""
import sys
import os

# Add frontend to Python path
sys.path.append('frontend')

# Import and run the main app
from streamlit_app import main

if __name__ == "__main__":
    main()