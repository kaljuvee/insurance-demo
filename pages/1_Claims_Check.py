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
if 'extracted_policy_data' not in st.session_state:
    st.session_state.extracted_policy_data = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Function to compare claim against policy
def compare_documents(claim_data, policy_data):
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Load the system prompt
    with open("prompts/claim_comparison.md", "r") as f:
        system_prompt = f.read()
    
    user_prompt = f"""
    Compare the following claim/invoice data against the policy contract data:
    
    CLAIM/INVOICE DATA:
    {json.dumps(claim_data, indent=2)}
    
    POLICY CONTRACT DATA:
    {json.dumps(policy_data, indent=2)}
    
    Provide a detailed comparison highlighting any discrepancies or issues.
    Return the results as a JSON object with the following structure:
    {{
        "document_summary": "Brief overview of the documents analyzed",
        "comparison_results": [
            {{
                "field": "Field name",
                "claim_value": "Value from claim",
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

# API Key input
with st.sidebar:
    st.header("OpenAI API Key")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API key set successfully!")
    else:
        st.warning("Please enter your OpenAI API key to use this application")

# Main content
if 'openai_api_key' in st.session_state:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Claim/Invoice")
        claim_file = st.file_uploader("Upload a claim or invoice PDF", type=["pdf"], key="claim_uploader")
        
        if claim_file:
            with st.spinner("Extracting text from claim/invoice..."):
                claim_text = extract_text_from_pdf(claim_file)
                st.session_state.claim_text = claim_text
                
            if st.button("Extract Claim Data"):
                with st.spinner("Analyzing claim with AI..."):
                    try:
                        extracted_claim_data = extract_data_with_openai(claim_text, "claim/invoice")
                        st.session_state.extracted_claim_data = extracted_claim_data
                        st.success("Claim data extracted successfully!")
                    except Exception as e:
                        st.error(f"Error extracting claim data: {str(e)}")
    
    with col2:
        st.header("Upload Policy Contract")
        policy_file = st.file_uploader("Upload a policy contract PDF", type=["pdf"], key="policy_uploader")
        
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
                    except Exception as e:
                        st.error(f"Error extracting policy data: {str(e)}")
    
    # Compare button
    if st.session_state.extracted_claim_data and st.session_state.extracted_policy_data:
        if st.button("Compare Documents"):
            with st.spinner("Comparing documents..."):
                try:
                    comparison_results = compare_documents(
                        st.session_state.extracted_claim_data,
                        st.session_state.extracted_policy_data
                    )
                    st.session_state.comparison_results = comparison_results
                    st.success("Comparison completed!")
                except Exception as e:
                    st.error(f"Error comparing documents: {str(e)}")
    
    # Display results
    if st.session_state.comparison_results:
        st.header("Comparison Results")
        
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
else:
    st.info("Please enter your OpenAI API key in the sidebar to use this application") 