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

st.set_page_config(
    page_title="Claims Check",
    page_icon="🔍",
    layout="wide"
)

st.title("Claims Check")
st.markdown("Upload invoice/claim documents and compare them against policy contracts")

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

# API Key input
with st.sidebar:
    st.header("OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your OpenAI API key to use this application")
    
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
        st.experimental_rerun()

# Main content
if 'openai_api_key' in st.session_state:
    # Create tabs for different sections
    tabs = st.tabs(["Document Analysis", "Image Analysis Integration", "Results"])
    
    with tabs[0]:
        # Demo data section
        if st.session_state.use_demo_data:
            st.info("Using demo data from the data directory")
            
            # Display demo files
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.header("Claim Document")
                st.markdown("Using: **insurance-claim-kristjan-tamm-001.pdf**")
                
                # Display a preview image
                st.image("https://raw.githubusercontent.com/kaljuvee/insurance-demo/main/data/claim_preview.png", 
                         caption="Claim Document Preview", 
                         use_column_width=True)
                
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
                
                # Display a preview image
                st.image("https://raw.githubusercontent.com/kaljuvee/insurance-demo/main/data/invoice_preview.png", 
                         caption="Invoice Document Preview", 
                         use_column_width=True)
                
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
                
                # Display a preview image
                st.image("https://raw.githubusercontent.com/kaljuvee/insurance-demo/main/data/policy_preview.png", 
                         caption="Policy Contract Preview", 
                         use_column_width=True)
                
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
                invoice_file = st.file_uploader("Upload an invoice document", type=["pdf"], key="invoice_uploader")
                
                if invoice_file:
                    with st.spinner("Extracting text from invoice..."):
                        invoice_text = extract_text_from_pdf(invoice_file)
                        st.session_state.invoice_text = invoice_text
                        
                    if st.button("Extract Invoice Data"):
                        with st.spinner("Analyzing invoice with AI..."):
                            try:
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
    
    with tabs[2]:
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
                
                st.dataframe(comparison_df.style.apply(highlight_mismatches, axis=1))
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
                
                st.dataframe(discrepancies_df.style.apply(highlight_severity, axis=1))
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
    st.info("Please enter your OpenAI API key in the sidebar to use this application") 