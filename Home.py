import streamlit as st
from utils import load_api_keys

# Load API keys from .env file
api_keys_loaded = load_api_keys()

st.set_page_config(
    page_title="Insurance Claims Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Insurance Claims Analyzer")

st.markdown("""
## Welcome to the Insurance Claims Analyzer

This application helps insurance professionals and policyholders analyze and compare insurance claims against policy contracts.

### Features:

- **Claims Check**: Upload invoice/claim documents and compare them against policy contracts
- **Image Detection**: Upload images of damaged property to identify features and potential damage
- **Automated Extraction**: Uses AI to extract relevant information from PDF documents and images
- **Side-by-Side Comparison**: Easily identify discrepancies between claims and policy coverage
- **Highlighted Differences**: Quickly spot potential issues or areas of concern

### How to Use:

1. Navigate to the desired page using the sidebar:
   - **Claims Check**: For document comparison and analysis
   - **Image Detection**: For analyzing images of property damage
2. Upload your documents or images
3. Select the appropriate analysis options
4. Review the detailed AI-powered analysis

### Benefits:

- Save time by automating document and image analysis
- Reduce errors in claims processing
- Ensure compliance with policy terms
- Improve customer satisfaction with faster claims resolution
- Document damage effectively with AI-powered image analysis

---
*This tool uses OpenAI and Anthropic language models to extract and analyze content.*
""")

# Display API key status in the sidebar
st.sidebar.title("Navigation")
st.sidebar.info("Select a page from the dropdown above")

# Show API key status
st.sidebar.markdown("---")
st.sidebar.header("API Key Status")

if api_keys_loaded['openai_api_key']:
    st.sidebar.success("✅ OpenAI API key loaded")
else:
    st.sidebar.warning("⚠️ OpenAI API key not found in .env file")

if api_keys_loaded['anthropic_api_key']:
    st.sidebar.success("✅ Anthropic API key loaded")
else:
    st.sidebar.info("ℹ️ Anthropic API key not found (optional)")

st.sidebar.markdown("""
### Setup Instructions

1. Create a `.env` file in the root directory
2. Add your API keys in the following format:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```
3. Restart the application if needed

You can also enter API keys directly in each tool page.
""") 