import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Documentation",
    page_icon="📘",
    layout="wide"
)

st.title("Documentation & User Guide")
st.markdown("Use this page to learn how to use the Insurance Claims Analyzer and download the full user guide.")

docs_dir = Path("docs")
md_path = docs_dir / "if-insurance-ai-demo-user-guide.md"
pdf_path = docs_dir / "if-insurance-ai-demo-user-guide.pdf"

col1, col2 = st.columns(2)
with col1:
    st.subheader("User Guide (Markdown)")
    if md_path.exists():
        st.success(f"Found {md_path.name}")
        st.download_button(
            label="Download Markdown",
            data=md_path.read_text(encoding="utf-8"),
            file_name=md_path.name,
            mime="text/markdown"
        )
    else:
        st.info("Markdown guide not found yet. It will be generated automatically.")

with col2:
    st.subheader("User Guide (PDF)")
    if pdf_path.exists():
        st.success(f"Found {pdf_path.name}")
        st.download_button(
            label="Download PDF",
            data=pdf_path.read_bytes(),
            file_name=pdf_path.name,
            mime="application/pdf"
        )
    else:
        st.warning("PDF guide not found yet. PDF will be created if a Markdown-to-PDF converter is available.")

st.markdown("---")
st.subheader("Quick Start")
st.markdown(
    """
    1. Add your API keys in `.env` or in the sidebar of each page.
    2. Go to `Claims Check` to extract data from claim, invoice, and policy documents (PDF or image invoice) and run comparison.
    3. Use `Fraud Detection` (in `Claims Check`) to evaluate any document in `data/` for manipulation or suspicious patterns.
    4. Go to `Image Detection` to analyze images for damage, fraud, or false positives/negatives. Save analyses and include them in comparisons.
    5. Export reports from each page as JSON or use this page to download the full user guide.
    """
)

st.markdown("---")
st.subheader("Regenerate Docs Locally")
st.markdown(
    """
    If you modify the Markdown and wish to regenerate the PDF:
    - Activate your virtual environment
    - Use a Markdown-to-PDF converter like `pandoc`

    Example:
    ```bash
    cd /home/julian/dev/hobby/insurance-demo
    source .venv/bin/activate
    pandoc docs/if-insurance-ai-demo-user-guide.md -o docs/if-insurance-ai-demo-user-guide.pdf
    ```
    """
)


