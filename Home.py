import streamlit as st

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
- **Automated Extraction**: Uses AI to extract relevant information from PDF documents
- **Side-by-Side Comparison**: Easily identify discrepancies between claims and policy coverage
- **Highlighted Differences**: Quickly spot potential issues or areas of concern

### How to Use:

1. Navigate to the **Claims Check** page using the sidebar
2. Upload your invoice/claim document
3. Upload or select the corresponding insurance policy contract
4. Review the side-by-side comparison and highlighted differences

### Benefits:

- Save time by automating document analysis
- Reduce errors in claims processing
- Ensure compliance with policy terms
- Improve customer satisfaction with faster claims resolution

---
*This tool uses OpenAI's language models to extract and analyze document content.*
""")

# Display sample images or icons in the sidebar
st.sidebar.title("Navigation")
st.sidebar.info("Select a page from the dropdown above") 