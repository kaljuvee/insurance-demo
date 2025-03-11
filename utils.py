import os
import streamlit as st
from dotenv import load_dotenv

def load_api_keys():
    """Load API keys from .env file and set them in session state"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Set API keys in session state if not already set
    if openai_api_key and 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = openai_api_key
    
    if anthropic_api_key and 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = anthropic_api_key
    
    return {
        'openai_api_key': openai_api_key is not None,
        'anthropic_api_key': anthropic_api_key is not None
    } 