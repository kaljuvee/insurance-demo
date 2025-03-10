# Insurance Claims Analyzer

A Streamlit application that helps insurance professionals and policyholders analyze and compare insurance claims against policy contracts using AI.

## Features

- **Claims Check**: Upload invoice/claim documents and compare them against policy contracts
- **Automated Extraction**: Uses OpenAI's language models to extract relevant information from PDF documents
- **Side-by-Side Comparison**: Easily identify discrepancies between claims and policy coverage
- **Highlighted Differences**: Quickly spot potential issues or areas of concern

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/insurance-demo.git
   cd insurance-demo
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run Home.py
   ```

## Usage

1. Start the application and navigate to the provided URL (typically http://localhost:8501)
2. Enter your OpenAI API key in the sidebar
3. Navigate to the "Claims Check" page using the sidebar
4. Upload your invoice/claim document
5. Upload the corresponding insurance policy contract
6. Click "Extract Claim Data" and "Extract Policy Data" to process the documents
7. Click "Compare Documents" to generate a detailed comparison
8. Review the side-by-side comparison and highlighted differences

## Sample Documents

The `data` directory contains sample insurance documents for testing:
- `invoice-kristjan-tamm-001.pdf` - Sample invoice
- `invoice-kristjan-tamm-002.pdf` - Another sample invoice
- `insurance-claim-kristjan-tamm-001.pdf` - Sample insurance claim
- `insurance-claim-kristjan-tamm-002.pdf` - Another sample insurance claim
- `insurance-contract-kristjan-tamm.pdf` - Sample insurance policy contract

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## How It Works

1. **PDF Text Extraction**: The application extracts text from uploaded PDF documents using PyPDF2
2. **AI-Powered Analysis**: OpenAI's language models extract structured data from the documents
3. **Intelligent Comparison**: The application compares the extracted data to identify discrepancies
4. **Visual Results**: Results are presented in an easy-to-understand format with highlighted differences

## License

This project is licensed under the terms of the license included in the repository.