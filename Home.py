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

This application helps insurance professionals and policyholders analyze and compare insurance claims against policy contracts with advanced fraud detection and accuracy assessment capabilities.

### Features:

- **Claims Check**: Upload invoice/claim documents and compare them against policy contracts
  - **NEW**: Support for image invoices (JPG, PNG, JPEG)
  - **NEW**: Fraud Detection step to identify doctored/fraudulent documents
  - **NEW**: Document selection from data folder in sidebar
- **Image Detection**: Upload images of damaged property to identify features and potential damage
  - **NEW**: Fraud Detection to identify manipulated or staged images
  - **NEW**: False Positive/Negative Detection with scoring system
  - **NEW**: Accuracy assessment for damage claims
- **Automated Extraction**: Uses AI to extract relevant information from PDF documents and images
- **Side-by-Side Comparison**: Easily identify discrepancies between claims and policy coverage
- **Highlighted Differences**: Quickly spot potential issues or areas of concern
- **Fraud Prevention**: Advanced AI-powered fraud detection for both documents and images

### How to Use:

1. Navigate to the desired page using the sidebar:
   - **Claims Check**: For document comparison, fraud detection, and analysis
   - **Image Detection**: For analyzing images of property damage with fraud detection
2. Upload your documents or images, or select from available documents in the data folder
3. Choose the appropriate analysis type (Comprehensive, Damage Assessment, Fraud Detection, etc.)
4. Review the detailed AI-powered analysis with risk scores and recommendations

### New Fraud Detection Capabilities:

- **Document Fraud Detection**: Analyzes PDFs and images for signs of manipulation, inconsistencies, and forgery
- **Image Fraud Detection**: Detects digital manipulation, staged damage, and suspicious patterns
- **False Positive/Negative Detection**: Identifies when clients present undamaged items as damaged or when damage might be overlooked
- **Risk Scoring**: Provides numerical risk scores and confidence levels for all assessments
- **Comprehensive Reporting**: Detailed reports with specific indicators and recommendations

### Benefits:

- Save time by automating document and image analysis
- Reduce errors in claims processing
- **NEW**: Prevent fraud with advanced detection capabilities
- **NEW**: Improve accuracy with false positive/negative detection
- Ensure compliance with policy terms
- Improve customer satisfaction with faster claims resolution
- Document damage effectively with AI-powered image analysis
- **NEW**: Reduce fraudulent claims and improve profitability

---
*This tool uses OpenAI and Anthropic language models to extract and analyze content with advanced fraud detection capabilities.*
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