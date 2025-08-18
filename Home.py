import streamlit as st
from utils import load_api_keys
from locale_util import init_language_selector, tr

# Load API keys from .env file
api_keys_loaded = load_api_keys()

st.set_page_config(
    page_title="Insurance Claims Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(tr("Insurance Claims Analyzer", "Kindlustusnõuete analüsaator"))

st.markdown(tr(
    """
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
""",
    """
## Tere tulemast kindlustusnõuete analüsaatorisse

See rakendus aitab kindlustusspetsialistidel ja kindlustusvõtjatel analüüsida ning võrrelda kindlustusnõudeid poliisilepingutega. Rakendusel on sisseehitatud pettuse tuvastamine ja täpsuse hindamine.

### Funktsioonid:

- **Nõuete kontroll**: Laadi üles arved/nõuded ja võrdle neid poliisilepingutega
  - **UUS**: Pildi-arvete tugi (JPG, PNG, JPEG)
  - **UUS**: Pettuse tuvastus samm, mis tuvastab võltsitud/manipuleeritud dokumendid
  - **UUS**: Külgribal dokumentide valik `data/` kaustast
- **Pildi tuvastus**: Laadi üles kahjustustega vara pildid ja tuvasta tunnused
  - **UUS**: Pettuse tuvastus manipuleeritud või lavastatud piltide avastamiseks
  - **UUS**: Vale-positiivsete/negatiivsete hindamine skooriga
  - **UUS**: Kahju tõsiduse hindamine
- **Automaatne väljavõte**: AI tuvastab PDFidest ja piltidest asjakohase info
- **Kõrvuti võrdlus**: Tuvasta hõlpsalt vastuolud nõuete ja poliiside vahel
- **Esiletõstetud erinevused**: Kiire ülevaade võimalikest probleemidest
- **Pettuste ennetamine**: Arvutipõhine pettuse tuvastus dokumentidele ja piltidele

### Kuidas kasutada:

1. Liigu külgriba abil sobivale lehele:
   - **Nõuete kontroll**: Dokumentide võrdlus, pettuse tuvastus
   - **Pildi tuvastus**: Kahjustuste analüüs piltidelt koos pettuse tuvastusega
2. Laadi üles dokumendid või pildid või vali need `data/` kaustast
3. Vali sobiv analüüsi tüüp (nt terviklik, kahju hindamine, pettuse tuvastus)
4. Vaata detailset AI-analüüsi koos riskiskooridega

### Uued pettuse tuvastuse võimalused:

- **Dokumentide pettuse tuvastus**: Tuvastab manipuleerimise, vastuolud ja võltsingu tunnused
- **Pildi pettuse tuvastus**: Tuvastab digitaalse manipuleerimise, lavastatud kahju ja ebajärjekindlused
- **Vale-positiivsed/negatiivsed**: Tuvastab ekslikud väited või tähelepanuta jäänud kahjud
- **Riskiskoorid**: Numbrilised riskid ja kindlustunne
- **Raportid**: Detailne ülevaade koos soovitustega

### Eelised:

- Säästab aega automatiseeritud analüüsiga
- Vähendab vigu menetluses
- **UUS**: Ennetab pettusi
- **UUS**: Parandab täpsust vale-positiivsete/negatiivsete tuvastusel
- Tagab vastavuse poliisi tingimustele
- Parandab kliendirahulolu
- Dokumenteerib kahju tõhusalt
- **UUS**: Vähendab pettusnõudeid

---
*See tööriist kasutab OpenAI ja Anthropic mudeleid pettuse tuvastuse ja analüüsi jaoks.*
"""
))

# Display API key status in the sidebar
st.sidebar.title(tr("Navigation", "Navigatsioon"))
init_language_selector()
st.sidebar.info(tr("Select a page from the dropdown above", "Vali leht ülalolevast rippmenüüst"))

# Show API key status
st.sidebar.markdown("---")
st.sidebar.header(tr("API Key Status", "API võtmete olek"))

if api_keys_loaded['openai_api_key']:
    st.sidebar.success(tr("✅ OpenAI API key loaded", "✅ OpenAI API võti leitud"))
else:
    st.sidebar.warning(tr("⚠️ OpenAI API key not found in .env file", "⚠️ OpenAI API võtit ei leitud .env failist"))

if api_keys_loaded['anthropic_api_key']:
    st.sidebar.success(tr("✅ Anthropic API key loaded", "✅ Anthropic API võti leitud"))
else:
    st.sidebar.info(tr("ℹ️ Anthropic API key not found (optional)", "ℹ️ Anthropic API võti puudub (valikuline)"))

st.sidebar.markdown(tr(
    """
### Setup Instructions

1. Create a `.env` file in the root directory
2. Add your API keys in the following format:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```
3. Restart the application if needed

You can also enter API keys directly in each tool page.
""",
    """
### Seadistamine

1. Loo projekti juurkausta `.env` fail
2. Lisa API võtmed järgmises vormingus:
   ```
   OPENAI_API_KEY=siia_sinu_openai_võti
   ANTHROPIC_API_KEY=siia_sinu_anthropic_võti
   ```
3. Taaskäivita rakendus vajadusel

Võid sisestada võtmed ka otse iga lehe külgribal.
"""
))