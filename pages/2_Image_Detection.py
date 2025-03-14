import streamlit as st
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import json
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from utils import load_api_keys

st.set_page_config(
    page_title="Image Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("Image Detection")
st.markdown("Upload images of damaged property to identify features and potential damage")

# Load API keys from .env file
api_keys_loaded = load_api_keys()

# Initialize session state variables if they don't exist
if 'image_analysis_results' not in st.session_state:
    st.session_state.image_analysis_results = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'damage_assessment' not in st.session_state:
    st.session_state.damage_assessment = None

def encode_image_to_base64(image_file):
    """Convert an image file to base64 encoding"""
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def analyze_image_with_openai(image_base64, prompt, structured_output=False):
    """Analyze image using OpenAI's vision model"""
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Load the system prompt
    with open("prompts/image_analysis.md", "r") as f:
        system_prompt = f.read()
    
    # Add structured output instruction if needed
    if structured_output:
        system_prompt += "\n\nReturn your analysis as a JSON object with the following structure:\n"
        system_prompt += """
        {
            "image_overview": "Brief description of what the image shows",
            "property_details": {
                "property_type": "Type of property",
                "visible_features": ["Feature 1", "Feature 2", ...],
                "environmental_context": "Description of surroundings"
            },
            "damage_assessment": {
                "damage_type": "Type of damage",
                "severity": "minor/moderate/severe",
                "affected_areas": ["Area 1", "Area 2", ...],
                "description": "Detailed description of damage"
            },
            "cause_analysis": "Potential causes of the damage",
            "safety_concerns": ["Concern 1", "Concern 2", ...],
            "documentation_quality": "Assessment of the image quality for documentation",
            "extracted_text": "Any text visible in the image",
            "recommendations": ["Recommendation 1", "Recommendation 2", ...]
        }
        """
    
    messages = [
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
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500,
        response_format={"type": "json_object"} if structured_output else None
    )
    
    content = response.choices[0].message.content
    
    if structured_output:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, return as text
            return {"error": "Failed to parse JSON response", "raw_content": content}
    else:
        return content

def analyze_image_with_anthropic(image_base64, prompt, structured_output=False):
    """Analyze image using Anthropic's Claude model"""
    client = Anthropic(api_key=st.session_state.anthropic_api_key)
    
    # Load the system prompt
    with open("prompts/image_analysis.md", "r") as f:
        system_prompt = f.read()
    
    # Add structured output instruction if needed
    if structured_output:
        system_prompt += "\n\nReturn your analysis as a JSON object with the following structure:\n"
        system_prompt += """
        {
            "image_overview": "Brief description of what the image shows",
            "property_details": {
                "property_type": "Type of property",
                "visible_features": ["Feature 1", "Feature 2", ...],
                "environmental_context": "Description of surroundings"
            },
            "damage_assessment": {
                "damage_type": "Type of damage",
                "severity": "minor/moderate/severe",
                "affected_areas": ["Area 1", "Area 2", ...],
                "description": "Detailed description of damage"
            },
            "cause_analysis": "Potential causes of the damage",
            "safety_concerns": ["Concern 1", "Concern 2", ...],
            "documentation_quality": "Assessment of the image quality for documentation",
            "extracted_text": "Any text visible in the image",
            "recommendations": ["Recommendation 1", "Recommendation 2", ...]
        }
        """
    
    full_prompt = f"{system_prompt}\n\n{prompt}"
    if structured_output:
        full_prompt += "\n\nPlease format your response as a valid JSON object according to the structure specified above."
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
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
    
    content = response.content[0].text
    
    if structured_output:
        try:
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                return {"error": "No JSON found in response", "raw_content": content}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_content": content}
    else:
        return content

def extract_text_from_image(image_base64, model="openai"):
    """Extract text from image using AI vision models"""
    prompt = """
    Focus exclusively on extracting all text visible in this image. 
    Format the text maintaining the original layout as much as possible.
    If there are any forms, tables, or structured data, preserve that structure.
    Do not include any analysis or commentary - only extract the text.
    """
    
    if model == "openai":
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR system. Extract all text visible in the image accurately."
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
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    else:
        client = Anthropic(api_key=st.session_state.anthropic_api_key)
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an expert OCR system. Extract all text visible in the image accurately.\n\n" + prompt
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

def analyze_damage_in_image(image_base64, model="openai"):
    """Perform detailed damage analysis on an image"""
    prompt = """
    Analyze this image specifically for property damage. Focus on:
    1. Type of damage (water, fire, impact, structural, etc.)
    2. Severity of damage (minor, moderate, severe)
    3. Affected components or areas
    4. Potential causes
    5. Safety concerns
    6. Repair recommendations
    
    Return your analysis as a structured JSON object.
    """
    
    if model == "openai":
        return analyze_image_with_openai(image_base64, prompt, structured_output=True)
    else:
        return analyze_image_with_anthropic(image_base64, prompt, structured_output=True)

def json_to_df(json_data, exclude_keys=None):
    """Convert JSON data to a pandas DataFrame for display"""
    if exclude_keys is None:
        exclude_keys = []
        
    # Flatten the JSON if it's nested
    flat_data = {}
    
    def flatten(data, prefix=""):
        if isinstance(data, dict):
            for key, value in data.items():
                if key in exclude_keys:
                    continue
                new_key = f"{prefix}{key}" if prefix else key
                if isinstance(value, (dict, list)) and not isinstance(value, str):
                    flatten(value, f"{new_key}.")
                else:
                    flat_data[new_key] = value
        elif isinstance(data, list) and not isinstance(data, str):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    flatten(item, f"{prefix}[{i}].")
                else:
                    flat_data[f"{prefix}[{i}]"] = item
    
    flatten(json_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(flat_data.items(), columns=["Field", "Value"])
    return df

# API Key input in sidebar
with st.sidebar:
    st.header("API Keys")
    
    # Show status of loaded API keys
    if api_keys_loaded['openai_api_key']:
        st.success("OpenAI API key loaded from .env file")
    else:
        st.warning("OpenAI API key not found in .env file")
        
    if api_keys_loaded['anthropic_api_key']:
        st.success("Anthropic API key loaded from .env file")
    else:
        st.info("Anthropic API key not found in .env file (optional)")
    
    # Optional override for API keys
    st.subheader("Override API Keys (Optional)")
    
    # OpenAI API Key override
    openai_api_key = st.text_input("Enter OpenAI API key to override", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("OpenAI API key override applied!")
    
    # Anthropic API Key override
    anthropic_api_key = st.text_input("Enter Anthropic API key to override", type="password")
    if anthropic_api_key:
        st.session_state.anthropic_api_key = anthropic_api_key
        st.success("Anthropic API key override applied!")
    
    st.markdown("---")
    
    # Model selection
    st.header("Model Selection")
    model_option = st.radio(
        "Choose AI model for analysis:",
        ["OpenAI GPT-4o", "Anthropic Claude 3 Opus"]
    )
    
    # If Anthropic is selected but no API key is available, show warning
    if model_option == "Anthropic Claude 3 Opus" and 'anthropic_api_key' not in st.session_state:
        st.warning("Anthropic API key is required for Claude model. Please add it to your .env file or enter it above.")
    
    # Analysis type
    st.header("Analysis Type")
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Comprehensive Analysis", "Damage Assessment", "OCR (Text Extraction)"]
    )
    
    # Output format
    st.header("Output Format")
    output_format = st.radio(
        "Choose output format:",
        ["Structured (JSON)", "Narrative"]
    )
    structured_output = output_format == "Structured (JSON)"

# Main content
if 'openai_api_key' in st.session_state:
    # Create tabs for different sections
    tabs = st.tabs(["Image Upload & Analysis", "Results", "Saved Analyses"])
    
    with tabs[0]:
        st.header("Upload Image")
        
        # Demo image option
        use_demo = st.checkbox("Use demo image")
        
        if use_demo:
            st.info("Using demo image of property damage")
            # Display demo image
            demo_image_path = "https://raw.githubusercontent.com/kaljuvee/insurance-demo/main/data/damage_demo.jpg"
            st.image(demo_image_path, caption="Demo Image - Property Damage", use_container_width=True)
            
            # Analyze button for demo image
            if st.button("Analyze Demo Image"):
                with st.spinner("Downloading and analyzing demo image..."):
                    try:
                        import requests
                        from io import BytesIO
                        
                        # Download the image
                        response = requests.get(demo_image_path)
                        image_bytes = BytesIO(response.content)
                        
                        # Convert to base64
                        image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
                        
                        # Check if selected model is available
                        if model_option == "Anthropic Claude 3 Opus" and 'anthropic_api_key' not in st.session_state:
                            st.error("Anthropic API key is required for Claude model. Please add it to your .env file or enter it in the sidebar.")
                        else:
                            # Perform analysis based on selected type
                            if analysis_type == "OCR (Text Extraction)":
                                model = "openai" if model_option == "OpenAI GPT-4o" else "anthropic"
                                result = extract_text_from_image(image_base64, model)
                                st.session_state.extracted_text = result
                                st.session_state.image_analysis_results = {"extracted_text": result}
                            elif analysis_type == "Damage Assessment":
                                model = "openai" if model_option == "OpenAI GPT-4o" else "anthropic"
                                result = analyze_damage_in_image(image_base64, model)
                                st.session_state.damage_assessment = result
                                st.session_state.image_analysis_results = result
                            else:  # Comprehensive Analysis
                                if model_option == "OpenAI GPT-4o":
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
                                    result = analyze_image_with_openai(image_base64, prompt, structured_output)
                                else:
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
                                    result = analyze_image_with_anthropic(image_base64, prompt, structured_output)
                                
                                st.session_state.image_analysis_results = result
                            
                            st.success("Analysis completed!")
                            # Switch to results tab
                            tabs[1].active = True
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
        else:
            # User upload
            uploaded_image = st.file_uploader("Upload an image of property damage", type=["jpg", "jpeg", "png"])
            
            if uploaded_image:
                # Display the uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Convert image to base64
                image_base64 = encode_image_to_base64(uploaded_image)
                
                # Analysis button
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Check if selected model is available
                            if model_option == "Anthropic Claude 3 Opus" and 'anthropic_api_key' not in st.session_state:
                                st.error("Anthropic API key is required for Claude model. Please add it to your .env file or enter it in the sidebar.")
                            else:
                                # Perform analysis based on selected type
                                if analysis_type == "OCR (Text Extraction)":
                                    model = "openai" if model_option == "OpenAI GPT-4o" else "anthropic"
                                    result = extract_text_from_image(image_base64, model)
                                    st.session_state.extracted_text = result
                                    st.session_state.image_analysis_results = {"extracted_text": result}
                                elif analysis_type == "Damage Assessment":
                                    model = "openai" if model_option == "OpenAI GPT-4o" else "anthropic"
                                    result = analyze_damage_in_image(image_base64, model)
                                    st.session_state.damage_assessment = result
                                    st.session_state.image_analysis_results = result
                                else:  # Comprehensive Analysis
                                    if model_option == "OpenAI GPT-4o":
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
                                        result = analyze_image_with_openai(image_base64, prompt, structured_output)
                                    else:
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
                                        result = analyze_image_with_anthropic(image_base64, prompt, structured_output)
                                    
                                    st.session_state.image_analysis_results = result
                                
                                st.success("Analysis completed!")
                                # Switch to results tab
                                tabs[1].active = True
                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
    
    with tabs[1]:
        st.header("Analysis Results")
        
        if st.session_state.image_analysis_results:
            # Create subtabs for different views of the results
            result_tabs = st.tabs(["Formatted Results", "Raw Data", "Visualizations"])
            
            with result_tabs[0]:
                if analysis_type == "OCR (Text Extraction)":
                    st.subheader("Extracted Text")
                    st.text_area("Text Content", st.session_state.extracted_text, height=400)
                elif analysis_type == "Damage Assessment" or structured_output:
                    # Display structured results
                    results = st.session_state.image_analysis_results
                    
                    if "image_overview" in results:
                        st.subheader("Image Overview")
                        st.write(results.get("image_overview", "No overview available"))
                    
                    if "property_details" in results:
                        st.subheader("Property Details")
                        prop_details = results.get("property_details", {})
                        st.write(f"**Property Type:** {prop_details.get('property_type', 'Not specified')}")
                        
                        if "visible_features" in prop_details and isinstance(prop_details["visible_features"], list):
                            st.write("**Visible Features:**")
                            for feature in prop_details["visible_features"]:
                                st.write(f"- {feature}")
                        
                        st.write(f"**Environmental Context:** {prop_details.get('environmental_context', 'Not specified')}")
                    
                    if "damage_assessment" in results:
                        st.subheader("Damage Assessment")
                        damage = results.get("damage_assessment", {})
                        
                        # Create columns for damage details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Damage Type:** {damage.get('damage_type', 'Not specified')}")
                            st.write(f"**Severity:** {damage.get('severity', 'Not specified')}")
                        
                        with col2:
                            if "affected_areas" in damage and isinstance(damage["affected_areas"], list):
                                st.write("**Affected Areas:**")
                                for area in damage["affected_areas"]:
                                    st.write(f"- {area}")
                        
                        st.write("**Description:**")
                        st.write(damage.get("description", "No detailed description available"))
                    
                    if "cause_analysis" in results:
                        st.subheader("Cause Analysis")
                        st.write(results.get("cause_analysis", "No cause analysis available"))
                    
                    if "safety_concerns" in results and isinstance(results["safety_concerns"], list):
                        st.subheader("Safety Concerns")
                        for concern in results["safety_concerns"]:
                            st.write(f"- {concern}")
                    
                    if "documentation_quality" in results:
                        st.subheader("Documentation Quality")
                        st.write(results.get("documentation_quality", "No assessment available"))
                    
                    if "extracted_text" in results and results["extracted_text"]:
                        st.subheader("Extracted Text")
                        st.text_area("Text Content", results["extracted_text"], height=200)
                    
                    if "recommendations" in results and isinstance(results["recommendations"], list):
                        st.subheader("Recommendations")
                        for i, rec in enumerate(results["recommendations"]):
                            st.write(f"{i+1}. {rec}")
                else:
                    # Display narrative results
                    st.markdown(st.session_state.image_analysis_results)
            
            with result_tabs[1]:
                if structured_output or analysis_type == "Damage Assessment":
                    # Display as DataFrame
                    st.subheader("Structured Data")
                    results_df = json_to_df(st.session_state.image_analysis_results, 
                                           exclude_keys=["raw_content"] if "raw_content" in st.session_state.image_analysis_results else [])
                    st.dataframe(results_df, use_container_width=True)
                
                # Display raw JSON or text
                st.subheader("Raw Results")
                if isinstance(st.session_state.image_analysis_results, dict):
                    st.json(st.session_state.image_analysis_results)
                else:
                    st.text_area("Raw Text", st.session_state.image_analysis_results, height=400)
            
            with result_tabs[2]:
                if analysis_type == "Damage Assessment" or (structured_output and "damage_assessment" in st.session_state.image_analysis_results):
                    st.subheader("Damage Severity Visualization")
                    
                    # Extract severity for visualization
                    severity = "Unknown"
                    if analysis_type == "Damage Assessment":
                        damage_data = st.session_state.image_analysis_results.get("damage_assessment", {})
                        severity = damage_data.get("severity", "Unknown").lower()
                    elif "damage_assessment" in st.session_state.image_analysis_results:
                        damage_data = st.session_state.image_analysis_results["damage_assessment"]
                        severity = damage_data.get("severity", "Unknown").lower()
                    
                    # Create severity gauge
                    severity_value = 0
                    if "minor" in severity:
                        severity_value = 33
                    elif "moderate" in severity:
                        severity_value = 66
                    elif "severe" in severity:
                        severity_value = 100
                    
                    # Display gauge chart
                    st.progress(severity_value / 100)
                    
                    # Add color-coded label
                    if severity_value <= 33:
                        st.markdown(f"<h3 style='color:green'>Minor Damage</h3>", unsafe_allow_html=True)
                    elif severity_value <= 66:
                        st.markdown(f"<h3 style='color:orange'>Moderate Damage</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color:red'>Severe Damage</h3>", unsafe_allow_html=True)
                else:
                    st.info("Visualizations are available for Damage Assessment or Structured Comprehensive Analysis")
            
            # Add export button
            st.subheader("Export Results")
            if isinstance(st.session_state.image_analysis_results, dict):
                export_data = json.dumps(st.session_state.image_analysis_results, indent=2)
                file_extension = "json"
                mime_type = "application/json"
            else:
                export_data = st.session_state.image_analysis_results
                file_extension = "txt"
                mime_type = "text/plain"
            
            if st.download_button(
                label="Export Results",
                data=export_data,
                file_name=f"image_analysis_results.{file_extension}",
                mime=mime_type
            ):
                st.success("Results exported successfully!")
            
            # Add save for claims button
            if st.button("Save Analysis for Claims"):
                if 'saved_analyses' not in st.session_state:
                    st.session_state.saved_analyses = []
                
                # Save the analysis with image reference
                analysis_entry = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_name": uploaded_image.name if not use_demo else "demo_damage_image.jpg",
                    "analysis_type": analysis_type,
                    "model_used": model_option,
                    "analysis_result": export_data
                }
                
                st.session_state.saved_analyses.append(analysis_entry)
                st.success(f"Analysis saved! You can access it in the Claims Check page or the Saved Analyses tab.")
    
    with tabs[2]:
        st.header("Saved Analyses")
        
        if 'saved_analyses' in st.session_state and len(st.session_state.saved_analyses) > 0:
            st.success(f"You have {len(st.session_state.saved_analyses)} saved image analyses")
            
            # Display saved analyses
            for i, analysis in enumerate(st.session_state.saved_analyses):
                with st.expander(f"Analysis {i+1}: {analysis['image_name']} ({analysis['timestamp']})"):
                    st.write(f"**Analysis Type:** {analysis['analysis_type']}")
                    st.write(f"**Model Used:** {analysis['model_used']}")
                    st.markdown("**Analysis Result:**")
                    
                    # Check if the result is JSON
                    try:
                        if isinstance(analysis['analysis_result'], str):
                            result_data = json.loads(analysis['analysis_result'])
                            st.json(result_data)
                        else:
                            st.markdown(analysis['analysis_result'])
                    except json.JSONDecodeError:
                        st.markdown(analysis['analysis_result'])
                    
                    # Add delete button
                    if st.button(f"Delete Analysis {i+1}", key=f"delete_{i}"):
                        st.session_state.saved_analyses.pop(i)
                        st.experimental_rerun()
        else:
            st.info("No saved analyses yet. Analyze an image and click 'Save Analysis for Claims' to save it here.")
else:
    st.warning("OpenAI API key is required. Please add it to your .env file or enter it in the sidebar.") 