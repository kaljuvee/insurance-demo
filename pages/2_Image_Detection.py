import streamlit as st
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import json
from openai import OpenAI
from anthropic import Anthropic
import pandas as pd

st.set_page_config(
    page_title="Image Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Image Detection")
st.markdown("Upload images of damaged property to identify features and potential damage")

# Initialize session state variables if they don't exist
if 'image_analysis_results' not in st.session_state:
    st.session_state.image_analysis_results = None

def encode_image_to_base64(image_file):
    """Convert an image file to base64 encoding"""
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def analyze_image_with_openai(image_base64, prompt):
    """Analyze image using OpenAI's vision model"""
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Load the system prompt
    with open("prompts/image_analysis.md", "r") as f:
        system_prompt = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def analyze_image_with_anthropic(image_base64, prompt):
    """Analyze image using Anthropic's Claude model"""
    client = Anthropic(api_key=st.session_state.anthropic_api_key)
    
    # Load the system prompt
    with open("prompts/image_analysis.md", "r") as f:
        system_prompt = f.read()
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}\n\n{prompt}"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    )
    
    return response.content[0].text

def extract_text_from_image_openai(image_base64):
    """Extract text from image using OpenAI's vision model"""
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an expert OCR system. Extract all text visible in the image accurately."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text visible in this image. Format the text maintaining the original layout as much as possible."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def parse_analysis_results(text):
    """Parse the analysis results into a structured format"""
    try:
        # Try to parse as JSON if the model returned JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If not JSON, return the raw text
        return {"raw_analysis": text}

# API Key input in sidebar
with st.sidebar:
    st.header("API Keys")
    
    # OpenAI API Key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("OpenAI API key set!")
    
    # Anthropic API Key
    anthropic_api_key = st.text_input("Enter your Anthropic API key (optional)", type="password")
    if anthropic_api_key:
        st.session_state.anthropic_api_key = anthropic_api_key
        st.success("Anthropic API key set!")
    
    st.markdown("---")
    
    # Model selection
    st.header("Model Selection")
    model_option = st.radio(
        "Choose AI model for analysis:",
        ["OpenAI GPT-4 Vision", "Anthropic Claude 3 Opus"]
    )
    
    # Analysis type
    st.header("Analysis Type")
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Damage Detection", "OCR (Text Extraction)", "Comprehensive Analysis"]
    )

# Main content
if 'openai_api_key' in st.session_state:
    # Image upload
    st.header("Upload Image")
    uploaded_image = st.file_uploader("Upload an image of property damage", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to base64
        image_base64 = encode_image_to_base64(uploaded_image)
        
        # Analysis button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Determine which prompt to use based on analysis type
                    if analysis_type == "Damage Detection":
                        prompt = """
                        Analyze this image for any visible property damage. Focus on:
                        1. Identifying all visible features in the image
                        2. Detailed description of any damage visible
                        3. Potential causes of the damage
                        4. Severity assessment (minor, moderate, severe)
                        5. Recommendations for repair or further inspection
                        
                        Format your response according to the output format in the system prompt.
                        """
                    elif analysis_type == "OCR (Text Extraction)":
                        prompt = """
                        Extract all text visible in this image. Format the text maintaining the original layout as much as possible.
                        If there are any forms, tables, or structured data, try to preserve that structure.
                        
                        Focus on the "Extracted Text" section of the output format.
                        """
                    else:  # Comprehensive Analysis
                        prompt = """
                        Provide a comprehensive analysis of this image including all sections specified in the output format:
                        1. Image Overview
                        2. Property Details
                        3. Damage Assessment
                        4. Cause Analysis
                        5. Severity Rating
                        6. Documentation Quality
                        7. Extracted Text
                        8. Recommendations
                        
                        Be thorough and detailed in your analysis.
                        """
                    
                    # Determine which model to use
                    if model_option == "OpenAI GPT-4 Vision" or 'anthropic_api_key' not in st.session_state:
                        analysis_result = analyze_image_with_openai(image_base64, prompt)
                    else:
                        analysis_result = analyze_image_with_anthropic(image_base64, prompt)
                    
                    # If OCR was selected, also perform text extraction
                    if analysis_type == "OCR (Text Extraction)" and model_option == "OpenAI GPT-4 Vision":
                        analysis_result = extract_text_from_image_openai(image_base64)
                    
                    # Store results in session state
                    st.session_state.image_analysis_results = analysis_result
                    st.success("Analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
        
        # Display analysis results
        if st.session_state.image_analysis_results:
            st.header("Analysis Results")
            
            # Create tabs for different views of the results
            tabs = st.tabs(["Formatted Results", "Raw Results"])
            
            with tabs[0]:
                st.markdown(st.session_state.image_analysis_results)
                
                # Add export button
                if st.download_button(
                    label="Export Results",
                    data=st.session_state.image_analysis_results,
                    file_name="image_analysis_results.txt",
                    mime="text/plain"
                ):
                    st.success("Results exported successfully!")
            
            with tabs[1]:
                st.text_area("Raw Results", st.session_state.image_analysis_results, height=400)
                
            # Add option to save analysis to claim
            st.subheader("Add to Claim")
            st.info("This analysis can be added to an existing claim in the Claims Check page.")
            if st.button("Save Analysis for Claims"):
                if 'saved_analyses' not in st.session_state:
                    st.session_state.saved_analyses = []
                
                # Save the analysis with image reference
                analysis_entry = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": uploaded_image.name,
                    "analysis_type": analysis_type,
                    "model_used": model_option,
                    "analysis_result": st.session_state.image_analysis_results
                }
                
                st.session_state.saved_analyses.append(analysis_entry)
                st.success(f"Analysis saved! You can access it in the Claims Check page.")
else:
    st.info("Please enter your OpenAI API key in the sidebar to use this application") 