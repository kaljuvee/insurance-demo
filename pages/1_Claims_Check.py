import streamlit as st
import os
import tempfile
import pandas as pd
import PyPDF2
import openai
from openai import OpenAI
import json
import re
from io import StringIO
from PIL import Image
from utils import load_api_keys
import base64
from pathlib import Path

st.set_page_config(
    page_title="Claims Check",
    page_icon="🔍",
    layout="wide"
)

st.title("Claims Check")
st.markdown("Upload invoice/claim documents and compare them against policy contracts")

# Load API keys from .env file
api_keys_loaded = load_api_keys()

# Initialize session state variables if they don't exist
if 'extracted_claim_data' not in st.session_state:
    st.session_state.extracted_claim_data = None
if 'extracted_invoice_data' not in st.session_state:
    st.session_state.extracted_invoice_data = None
if 'extracted_policy_data' not in st.session_state:
    st.session_state.extracted_policy_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'saved_analyses' not in st.session_state:
    st.session_state.saved_analyses = []
if 'use_demo_data' not in st.session_state:
    st.session_state.use_demo_data = False
if 'fraud_detection_results' not in st.session_state:
    st.session_state.fraud_detection_results = None

# Function to get all documents from data folder
def get_data_folder_documents():
    """Get all documents from the data folder and subfolders"""
    documents = []
    data_path = Path("data")
    
    if data_path.exists():
        # Get all files recursively
        for file_path in data_path.rglob("*"):
            if file_path.is_file():
                # Get relative path from data folder
                rel_path = file_path.relative_to(data_path)
                documents.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": file_path.suffix.lower(),
                    "relative_path": str(rel_path)
                })
    
    return documents

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    text = ""
    with open(temp_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    
    os.unlink(temp_path)
    return text

# Function to extract text from PDF file path
def extract_text_from_pdf_path(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    
    return text

# Function to encode image to base64
def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to detect fraud in documents
def detect_fraud_document(document_text, document_type, image_base64=None):
    """Detect potential fraud in documents using AI"""
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    system_prompt = """
    You are an expert fraud detection specialist for insurance claims. Analyze the provided document for potential fraud indicators.
    
    Look for:
    1. Inconsistencies in dates, amounts, or details
    2. Signs of document manipulation or forgery
    3. Unusual patterns or suspicious information
    4. Mismatches between different parts of the document
    5. Signs of digital manipulation in images
    6. Inconsistent formatting or typography
    7. Suspicious metadata or timestamps
    
    Return your analysis as a JSON object with the following structure:
    {
        "fraud_risk_score": 0-100,
        "risk_level": "low/medium/high",
        "fraud_indicators": [
            {
                "indicator": "Description of the fraud indicator",
                "severity": "low/medium/high",
                "confidence": 0-100,
                "explanation": "Detailed explanation"
            }
        ],
        "document_authenticity": "authentic/suspicious/fraudulent",
        "recommendations": [
            "Recommendation 1",
            "Recommendation 2"
        ],
        "overall_assessment": "Detailed overall assessment"
    }
    """
    
    user_prompt = f"""
    Analyze this {document_type} for potential fraud indicators.
    
    Document text:
    {document_text}
    
    Provide a comprehensive fraud detection analysis.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # If image is provided, add it to the analysis
    if image_base64:
        messages[1]["content"] += f"\n\nAn image version of this document is also available for visual analysis."
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Please also analyze this image for visual signs of fraud, manipulation, or forgery."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Function to extract data using OpenAI
def extract_data_with_openai(text, document_type):
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Load the system prompt
    with open("prompts/claim_comparison.md", "r") as f:
        system_prompt = f.read()
    
    user_prompt = f"""
    Extract all relevant information from this {document_type} document. 
    The document text is provided below:
    
    {text}
    
    Return the extracted information as a JSON object with all relevant fields.
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Function to compare claim against policy
def compare_documents(claim_data, invoice_data, policy_data, image_analyses=None):
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Load the system prompt
    with open("prompts/claim_comparison.md", "r") as f:
        system_prompt = f.read()
    
    # Prepare image analyses text if available
    image_analyses_text = ""
    if image_analyses and len(image_analyses) > 0:
        image_analyses_text = "IMAGE ANALYSES:\n"
        for i, analysis in enumerate(image_analyses):
            image_analyses_text += f"\nImage {i+1}: {analysis['image_name']}\n"
            image_analyses_text += f"Analysis Type: {analysis['analysis_type']}\n"
            image_analyses_text += f"Model Used: {analysis['model_used']}\n"
            image_analyses_text += f"Analysis Result:\n{analysis['analysis_result']}\n"
            image_analyses_text += "-" * 50 + "\n"
    
    user_prompt = f"""
    Compare the following claim and invoice data against the policy contract data:
    
    CLAIM DATA:
    {json.dumps(claim_data, indent=2)}
    
    INVOICE DATA:
    {json.dumps(invoice_data, indent=2)}
    
    POLICY CONTRACT DATA:
    {json.dumps(policy_data, indent=2)}
    
    {image_analyses_text}
    
    Provide a detailed comparison highlighting any discrepancies or issues.
    Return the results as a JSON object with the following structure:
    {{
        "document_summary": "Brief overview of the documents analyzed",
        "comparison_results": [
            {{
                "field": "Field name",
                "claim_value": "Value from claim",
                "invoice_value": "Value from invoice",
                "policy_value": "Value from policy",
                "match": true/false,
                "notes": "Any notes about this comparison"
            }}
        ],
        "discrepancies": [
            {{
                "field": "Field with discrepancy",
                "description": "Description of the issue",
                "severity": "high/medium/low"
            }}
        ],
        "recommendations": [
            "Recommendation 1",
            "Recommendation 2"
        ]
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Function to convert JSON to DataFrame
def json_to_df(json_data):
    """Convert JSON data to a pandas DataFrame for display"""
    # Flatten the JSON if it's nested
    flat_data = {}
    
    def flatten(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, (dict, list)) and not isinstance(value, str):
                    flatten(value, f"{new_key}.")
                else:
                    flat_data[new_key] = value
        elif isinstance(data, list) and not isinstance(data, str):
            for i, item in enumerate(data):
                flatten(item, f"{prefix}[{i}].")
    
    flatten(json_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(flat_data.items(), columns=["Field", "Value"])
    return df

# Sidebar for document selection
with st.sidebar:
    st.header("Document Selection")
    
    # Get all documents from data folder
    data_documents = get_data_folder_documents()
    
    if data_documents:
        st.subheader("Available Documents")
        
        # Group documents by type
        pdfs = [doc for doc in data_documents if doc["type"] == ".pdf"]
        images = [doc for doc in data_documents if doc["type"] in [".jpg", ".jpeg", ".png"]]
        
        # Document type selection
        doc_type = st.selectbox(
            "Select document type:",
            ["PDF Documents", "Image Documents", "All Documents"]
        )
        
        if doc_type == "PDF Documents":
            documents_to_show = pdfs
        elif doc_type == "Image Documents":
            documents_to_show = images
        else:
            documents_to_show = data_documents
        
        # Document selection
        if documents_to_show:
            selected_doc = st.selectbox(
                "Select a document:",
                documents_to_show,
                format_func=lambda x: f"{x['name']} ({x['relative_path']})"
            )
            
            if selected_doc:
                st.info(f"Selected: {selected_doc['name']}")
                st.write(f"Path: {selected_doc['relative_path']}")
                st.write(f"Type: {selected_doc['type']}")
                
                # Store selected document in session state
                st.session_state.selected_document = selected_doc
        else:
            st.warning(f"No {doc_type.lower()} found in data folder")
    else:
        st.warning("No documents found in data folder")

# API Key input
with st.sidebar:
    st.header("API Keys")
    
    # Show status of loaded API keys
    if api_keys_loaded['openai_api_key']:
        st.success("OpenAI API key loaded from .env file")
    else:
        st.warning("OpenAI API key not found in .env file")
    
    # Optional override for OpenAI API key
    st.subheader("Override API Key (Optional)")
    openai_api_key = st.text_input("Enter OpenAI API key to override", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API key override applied!")
    
    # Demo data checkbox
    st.header("Demo Options")
    use_demo = st.checkbox("Use demo data", value=st.session_state.use_demo_data)
    if use_demo != st.session_state.use_demo_data:
        st.session_state.use_demo_data = use_demo
        # Reset extracted data when toggling demo mode
        st.session_state.extracted_claim_data = None
        st.session_state.extracted_invoice_data = None
        st.session_state.extracted_policy_data = None
        st.session_state.comparison_results = None
        st.rerun()

# Main content
if 'openai_api_key' in st.session_state:
    # Create tabs for different sections
    tabs = st.tabs(["Document Analysis", "Fraud Detection", "Image Analysis Integration", "Compare"])
    
    with tabs[0]:
        # Demo data section
        if st.session_state.use_demo_data:
            st.info("Using demo data from the data directory")
            
            # Display demo files
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.header("Claim Document")
                st.markdown("Using: **insurance-claim-kristjan-tamm-001.pdf**")
                
                if st.button("Extract Claim Data"):
                    with st.spinner("Analyzing claim with AI..."):
                        try:
                            claim_text = extract_text_from_pdf_path("data/insurance-claim-kristjan-tamm-001.pdf")
                            extracted_claim_data = extract_data_with_openai(claim_text, "claim")
                            st.session_state.extracted_claim_data = extracted_claim_data
                            st.success("Claim data extracted successfully!")
                            
                            # Display extracted data as DataFrame
                            st.subheader("Extracted Claim Data")
                            claim_df = json_to_df(extracted_claim_data)
                            st.dataframe(claim_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error extracting claim data: {str(e)}")
            
            with col2:
                st.header("Invoice Document")
                st.markdown("Using: **invoice-kristjan-tamm-001.pdf**")
                
                if st.button("Extract Invoice Data"):
                    with st.spinner("Analyzing invoice with AI..."):
                        try:
                            invoice_text = extract_text_from_pdf_path("data/invoice-kristjan-tamm-001.pdf")
                            extracted_invoice_data = extract_data_with_openai(invoice_text, "invoice")
                            st.session_state.extracted_invoice_data = extracted_invoice_data
                            st.success("Invoice data extracted successfully!")
                            
                            # Display extracted data as DataFrame
                            st.subheader("Extracted Invoice Data")
                            invoice_df = json_to_df(extracted_invoice_data)
                            st.dataframe(invoice_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error extracting invoice data: {str(e)}")
            
            with col3:
                st.header("Policy Contract")
                st.markdown("Using: **insurance-contract-kristjan-tamm.pdf**")
                
                if st.button("Extract Policy Data"):
                    with st.spinner("Analyzing policy with AI..."):
                        try:
                            policy_text = extract_text_from_pdf_path("data/insurance-contract-kristjan-tamm.pdf")
                            extracted_policy_data = extract_data_with_openai(policy_text, "policy contract")
                            st.session_state.extracted_policy_data = extracted_policy_data
                            st.success("Policy data extracted successfully!")
                            
                            # Display extracted data as DataFrame
                            st.subheader("Extracted Policy Data")
                            policy_df = json_to_df(extracted_policy_data)
                            st.dataframe(policy_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error extracting policy data: {str(e)}")
        
        # User upload section
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.header("Upload Claim")
                claim_file = st.file_uploader("Upload a claim document", type=["pdf"], key="claim_uploader")
                
                if claim_file:
                    with st.spinner("Extracting text from claim..."):
                        claim_text = extract_text_from_pdf(claim_file)
                        st.session_state.claim_text = claim_text
                        
                    if st.button("Extract Claim Data"):
                        with st.spinner("Analyzing claim with AI..."):
                            try:
                                extracted_claim_data = extract_data_with_openai(claim_text, "claim")
                                st.session_state.extracted_claim_data = extracted_claim_data
                                st.success("Claim data extracted successfully!")
                                
                                # Display extracted data as DataFrame
                                st.subheader("Extracted Claim Data")
                                claim_df = json_to_df(extracted_claim_data)
                                st.dataframe(claim_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error extracting claim data: {str(e)}")
            
            with col2:
                st.header("Upload Invoice")
                st.markdown("**Support for both PDF and Image formats**")
                invoice_file = st.file_uploader("Upload an invoice document", type=["pdf", "jpg", "jpeg", "png"], key="invoice_uploader")
                
                if invoice_file:
                    if invoice_file.type == "application/pdf":
                        with st.spinner("Extracting text from invoice PDF..."):
                            invoice_text = extract_text_from_pdf(invoice_file)
                            st.session_state.invoice_text = invoice_text
                    else:
                        # Handle image invoice
                        st.image(invoice_file, caption="Invoice Image", use_container_width=True)
                        st.info("Image invoice detected. Use the Fraud Detection tab to analyze this image.")
                        invoice_text = "Image invoice - text extraction not available"
                        st.session_state.invoice_text = invoice_text
                        st.session_state.invoice_image = invoice_file
                        
                    if st.button("Extract Invoice Data"):
                        with st.spinner("Analyzing invoice with AI..."):
                            try:
                                if invoice_file.type == "application/pdf":
                                    extracted_invoice_data = extract_data_with_openai(invoice_text, "invoice")
                                else:
                                    # For image invoices, create a basic structure
                                    extracted_invoice_data = {
                                        "document_type": "invoice_image",
                                        "note": "Image invoice - detailed extraction requires visual analysis",
                                        "filename": invoice_file.name
                                    }
                                st.session_state.extracted_invoice_data = extracted_invoice_data
                                st.success("Invoice data extracted successfully!")
                                
                                # Display extracted data as DataFrame
                                st.subheader("Extracted Invoice Data")
                                invoice_df = json_to_df(extracted_invoice_data)
                                st.dataframe(invoice_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error extracting invoice data: {str(e)}")
            
            with col3:
                st.header("Upload Policy Contract")
                policy_file = st.file_uploader("Upload a policy contract", type=["pdf"], key="policy_uploader")
                
                if policy_file:
                    with st.spinner("Extracting text from policy contract..."):
                        policy_text = extract_text_from_pdf(policy_file)
                        st.session_state.policy_text = policy_text
                        
                    if st.button("Extract Policy Data"):
                        with st.spinner("Analyzing policy with AI..."):
                            try:
                                extracted_policy_data = extract_data_with_openai(policy_text, "policy contract")
                                st.session_state.extracted_policy_data = extracted_policy_data
                                st.success("Policy data extracted successfully!")
                                
                                # Display extracted data as DataFrame
                                st.subheader("Extracted Policy Data")
                                policy_df = json_to_df(extracted_policy_data)
                                st.dataframe(policy_df, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error extracting policy data: {str(e)}")
    
    with tabs[1]:
        st.header("Fraud Detection")
        st.markdown("Analyze documents for potential fraud indicators")
        
        # Check if we have documents to analyze
        if hasattr(st.session_state, 'selected_document') and st.session_state.selected_document:
            selected_doc = st.session_state.selected_document
            
            st.subheader(f"Analyzing: {selected_doc['name']}")
            
            # Display document info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Document Type:** {selected_doc['type']}")
                st.write(f"**Path:** {selected_doc['relative_path']}")
            
            with col2:
                if selected_doc['type'] in ['.jpg', '.jpeg', '.png']:
                    # Display image
                    st.image(selected_doc['path'], caption=selected_doc['name'], use_container_width=True)
            
            # Fraud detection button
            if st.button("Detect Fraud"):
                with st.spinner("Analyzing document for fraud indicators..."):
                    try:
                        # Extract text if it's a PDF
                        if selected_doc['type'] == '.pdf':
                            document_text = extract_text_from_pdf_path(selected_doc['path'])
                            image_base64 = None
                        else:
                            # For images, we'll analyze the image directly
                            document_text = "Image document - visual analysis required"
                            image_base64 = encode_image_to_base64(selected_doc['path'])
                        
                        # Perform fraud detection
                        fraud_results = detect_fraud_document(
                            document_text, 
                            selected_doc['type'], 
                            image_base64
                        )
                        
                        st.session_state.fraud_detection_results = fraud_results
                        st.success("Fraud detection completed!")
                        
                    except Exception as e:
                        st.error(f"Error during fraud detection: {str(e)}")
        
        # Display fraud detection results
        if st.session_state.fraud_detection_results:
            results = st.session_state.fraud_detection_results
            
            # Risk score visualization
            st.subheader("Fraud Risk Assessment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_score = results.get('fraud_risk_score', 0)
                st.metric("Risk Score", f"{risk_score}/100")
                
                # Progress bar for risk score
                st.progress(risk_score / 100)
            
            with col2:
                risk_level = results.get('risk_level', 'unknown').upper()
                if risk_level == 'LOW':
                    st.success(f"Risk Level: {risk_level}")
                elif risk_level == 'MEDIUM':
                    st.warning(f"Risk Level: {risk_level}")
                else:
                    st.error(f"Risk Level: {risk_level}")
            
            with col3:
                authenticity = results.get('document_authenticity', 'unknown').upper()
                if authenticity == 'AUTHENTIC':
                    st.success(f"Authenticity: {authenticity}")
                elif authenticity == 'SUSPICIOUS':
                    st.warning(f"Authenticity: {authenticity}")
                else:
                    st.error(f"Authenticity: {authenticity}")
            
            # Fraud indicators
            st.subheader("Fraud Indicators")
            if 'fraud_indicators' in results and results['fraud_indicators']:
                for i, indicator in enumerate(results['fraud_indicators']):
                    with st.expander(f"Indicator {i+1}: {indicator['indicator']}"):
                        st.write(f"**Severity:** {indicator['severity'].upper()}")
                        st.write(f"**Confidence:** {indicator['confidence']}%")
                        st.write(f"**Explanation:** {indicator['explanation']}")
                        
                        # Color code based on severity
                        if indicator['severity'].lower() == 'high':
                            st.error("⚠️ High Risk Indicator")
                        elif indicator['severity'].lower() == 'medium':
                            st.warning("⚠️ Medium Risk Indicator")
                        else:
                            st.info("ℹ️ Low Risk Indicator")
            else:
                st.success("No fraud indicators detected")
            
            # Overall assessment
            st.subheader("Overall Assessment")
            st.write(results.get('overall_assessment', 'No assessment available'))
            
            # Recommendations
            st.subheader("Recommendations")
            if 'recommendations' in results and results['recommendations']:
                for i, rec in enumerate(results['recommendations']):
                    st.write(f"{i+1}. {rec}")
            else:
                st.write("No recommendations available")
            
            # Export results
            st.subheader("Export Fraud Detection Report")
            if st.download_button(
                label="Download Fraud Detection Report",
                data=json.dumps(results, indent=2),
                file_name="fraud_detection_report.json",
                mime="application/json"
            ):
                st.success("Fraud detection report downloaded successfully!")
        
        else:
            st.info("Select a document from the sidebar and click 'Detect Fraud' to analyze it for potential fraud indicators.")
    
    with tabs[2]:
        st.header("Image Analysis Integration")
        
        if len(st.session_state.saved_analyses) > 0:
            st.success(f"You have {len(st.session_state.saved_analyses)} saved image analyses available")
            
            # Display saved analyses
            for i, analysis in enumerate(st.session_state.saved_analyses):
                with st.expander(f"Image Analysis {i+1}: {analysis['image_name']} ({analysis['timestamp']})"):
                    st.write(f"**Analysis Type:** {analysis['analysis_type']}")
                    st.write(f"**Model Used:** {analysis['model_used']}")
                    st.markdown("**Analysis Result:**")
                    st.markdown(analysis['analysis_result'])
            
            # Select analyses to include
            st.subheader("Select Analyses to Include in Claim Comparison")
            selected_analyses = []
            for i, analysis in enumerate(st.session_state.saved_analyses):
                if st.checkbox(f"Include {analysis['image_name']} in comparison", key=f"include_analysis_{i}"):
                    selected_analyses.append(analysis)
            
            st.session_state.selected_analyses = selected_analyses
            
            if len(selected_analyses) > 0:
                st.success(f"Selected {len(selected_analyses)} analyses for inclusion in claim comparison")
        else:
            st.info("No saved image analyses available. Go to the Image Detection page to analyze images and save the results.")
            st.markdown("""
            ### How to add image analyses:
            1. Navigate to the **Image Detection** page
            2. Upload and analyze an image
            3. Click the **Save Analysis for Claims** button
            4. Return to this page to include the analysis in your claim comparison
            """)
    
    with tabs[3]:
        st.header("Comparison Results")
        
        # Compare button
        if (st.session_state.extracted_claim_data and 
            st.session_state.extracted_invoice_data and 
            st.session_state.extracted_policy_data):
            
            selected_analyses = st.session_state.get('selected_analyses', [])
            include_images = len(selected_analyses) > 0
            
            compare_button_text = "Compare Documents"
            if include_images:
                compare_button_text += f" (Including {len(selected_analyses)} Image Analyses)"
                
            if st.button(compare_button_text):
                with st.spinner("Comparing documents..."):
                    try:
                        comparison_results = compare_documents(
                            st.session_state.extracted_claim_data,
                            st.session_state.extracted_invoice_data,
                            st.session_state.extracted_policy_data,
                            selected_analyses if include_images else None
                        )
                        st.session_state.comparison_results = comparison_results
                        st.success("Comparison completed!")
                    except Exception as e:
                        st.error(f"Error comparing documents: {str(e)}")
        else:
            missing = []
            if not st.session_state.extracted_claim_data:
                missing.append("Claim")
            if not st.session_state.extracted_invoice_data:
                missing.append("Invoice")
            if not st.session_state.extracted_policy_data:
                missing.append("Policy Contract")
            
            if missing:
                st.warning(f"Please extract data from all documents before comparing. Missing: {', '.join(missing)}")
        
        # Display results
        if st.session_state.comparison_results:
            # Document Summary
            st.subheader("Document Summary")
            st.write(st.session_state.comparison_results.get("document_summary", "No summary available"))
            
            # Comparison Table
            st.subheader("Side-by-Side Comparison")
            if "comparison_results" in st.session_state.comparison_results:
                comparison_df = pd.DataFrame(st.session_state.comparison_results["comparison_results"])
                
                # Apply styling to highlight mismatches
                def highlight_mismatches(row):
                    if not row['match']:
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(comparison_df.style.apply(highlight_mismatches, axis=1), use_container_width=True)
            else:
                st.write("No comparison data available")
            
            # Discrepancies
            st.subheader("Discrepancies")
            if "discrepancies" in st.session_state.comparison_results and st.session_state.comparison_results["discrepancies"]:
                discrepancies_df = pd.DataFrame(st.session_state.comparison_results["discrepancies"])
                
                # Apply styling based on severity
                def highlight_severity(row):
                    if row['severity'].lower() == 'high':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['severity'].lower() == 'medium':
                        return ['background-color: #ffffcc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(discrepancies_df.style.apply(highlight_severity, axis=1), use_container_width=True)
            else:
                st.write("No discrepancies found")
            
            # Recommendations
            st.subheader("Recommendations")
            if "recommendations" in st.session_state.comparison_results:
                for i, rec in enumerate(st.session_state.comparison_results["recommendations"]):
                    st.write(f"{i+1}. {rec}")
            else:
                st.write("No recommendations available")
            
            # Export results
            st.subheader("Export Results")
            if st.download_button(
                label="Download Comparison Report",
                data=json.dumps(st.session_state.comparison_results, indent=2),
                file_name="claim_comparison_report.json",
                mime="application/json"
            ):
                st.success("Report downloaded successfully!")
else:
    st.warning("OpenAI API key is required. Please add it to your .env file or enter it in the sidebar.") 