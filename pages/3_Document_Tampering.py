import streamlit as st
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import PyPDF2
import subprocess
import tempfile
from PIL import Image, ExifTags
from openai import OpenAI
from utils import load_api_keys
from locale_util import init_language_selector

st.set_page_config(
    page_title="Document Tampering Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("Document Tampering Detection")
st.markdown("Analyze metadata for PDFs and images, flag anomalies, and run AI-assisted tampering assessment.")

# Load keys
api_keys_loaded = load_api_keys()

# Session state
if 'tampering_results' not in st.session_state:
    st.session_state.tampering_results = None
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None


def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_pdf_date(date_str: str) -> datetime | None:
    # PDF dates like D:YYYYMMDDHHmmSSOHH'mm'
    if not date_str:
        return None
    s = date_str
    if s.startswith('D:'):
        s = s[2:]
    # Keep only digits for base parse
    try:
        # Try with timezone-less first
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        return dt
    except Exception:
        try:
            # Fallback to date only
            dt = datetime.strptime(s[:8], "%Y%m%d")
            return dt
        except Exception:
            return None


def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        info = reader.metadata or {}
        # Convert to regular dict with clean keys
        for k, v in info.items():
            key = str(k).lstrip('/')
            meta[key] = str(v)
        meta['num_pages'] = len(reader.pages)
    return meta


def extract_image_metadata(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    with Image.open(file_path) as img:
        meta['format'] = img.format
        meta['mode'] = img.mode
        meta['size'] = {'width': img.size[0], 'height': img.size[1]}
        if 'dpi' in img.info:
            meta['dpi'] = img.info.get('dpi')
        # EXIF
        exif_data = {}
        try:
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    # Ensure JSON-serializable
                    try:
                        exif_data[str(tag)] = value if isinstance(value, (int, float, str)) else str(value)
                    except Exception:
                        exif_data[str(tag)] = str(value)
        except Exception:
            pass
        meta['exif'] = exif_data
        # Additional info (PNG text etc.)
        extra_info = {}
        for k, v in img.info.items():
            if k != 'exif':
                extra_info[k] = v if isinstance(v, (int, float, str)) else str(v)
        if extra_info:
            meta['extra_info'] = extra_info
    return meta


def run_heuristics(document_type: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    # Common
    if not metadata or len(metadata) == 0:
        findings.append({
            'indicator': 'Metadata missing or empty',
            'severity': 'medium',
            'explanation': 'Document has no accessible metadata; while not conclusive, it can hinder authenticity checks.'
        })

    if document_type == 'pdf':
        creation = parse_pdf_date(metadata.get('CreationDate', ''))
        mod = parse_pdf_date(metadata.get('ModDate', ''))
        if creation and mod and mod < creation:
            findings.append({
                'indicator': 'Modification date precedes creation date',
                'severity': 'high',
                'explanation': 'Inconsistent timestamps can indicate manual metadata editing or file tampering.'
            })
        producer = (metadata.get('Producer') or '').lower()
        if any(tool in producer for tool in ['photoshop', 'gimp', 'image editor']):
            findings.append({
                'indicator': 'PDF producer suggests raster editor',
                'severity': 'medium',
                'explanation': 'PDF appears generated/processed by an image editor, which may be unusual for invoices or contracts.'
            })
        if not any(k in metadata for k in ['Author', 'Creator', 'Producer', 'Title']):
            findings.append({
                'indicator': 'Minimal descriptive fields',
                'severity': 'low',
                'explanation': 'Missing typical fields (Author/Creator/Title) can be a weak signal of sanitization.'
            })

    else:  # image
        exif = metadata.get('exif', {})
        software = (exif.get('Software') or metadata.get('extra_info', {}).get('Software') or '').lower()
        if any(s in software for s in ['photoshop', 'gimp', 'snapseed', 'lightroom', 'picsart']):
            findings.append({
                'indicator': 'Edited with image manipulation software',
                'severity': 'medium',
                'explanation': 'Presence of editing software in EXIF often indicates post-processing.'
            })
        dt_orig = exif.get('DateTimeOriginal')
        dt_mod = exif.get('DateTime') or exif.get('ModifyDate')
        try:
            if dt_orig and dt_mod and str(dt_mod) < str(dt_orig):
                findings.append({
                    'indicator': 'Modify time precedes original time',
                    'severity': 'medium',
                    'explanation': 'Inconsistent EXIF times can indicate manual manipulation.'
                })
        except Exception:
            pass
        if not exif:
            findings.append({
                'indicator': 'No EXIF data',
                'severity': 'low',
                'explanation': 'Some formats strip EXIF; absence isn’t conclusive but reduces provenance confidence.'
            })
    return findings


def augment_heuristics_with_external(document_type: str, exiftool_data: Dict[str, Any], qpdf_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    extra: List[Dict[str, Any]] = []
    if document_type == 'pdf' and qpdf_result:
        if qpdf_result.get('has_errors'):
            extra.append({
                'indicator': 'QPDF structural errors',
                'severity': 'high',
                'explanation': 'qpdf --check reported errors indicating potential PDF corruption or tampering.'
            })
        elif qpdf_result.get('has_warnings'):
            extra.append({
                'indicator': 'QPDF warnings',
                'severity': 'medium',
                'explanation': 'qpdf --check reported warnings; review structure and incremental updates.'
            })
    if document_type != 'pdf' and isinstance(exiftool_data, dict):
        sw = str(exiftool_data.get('Software', '')).lower()
        if any(s in sw for s in ['photoshop', 'gimp', 'snapseed', 'lightroom', 'picsart']):
            extra.append({
                'indicator': 'EXIF Software indicates editing tool',
                'severity': 'medium',
                'explanation': 'EXIF Software field shows common image editors; inspect for manipulation.'
            })
    return extra

def run_exiftool(file_path: str) -> Dict[str, Any]:
    """Run exiftool and return parsed JSON for both images and PDFs."""
    try:
        proc = subprocess.run(
            ["exiftool", "-json", "-n", file_path],
            capture_output=True,
            text=True,
            check=False
        )
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            return data[0] if isinstance(data, list) and data else {}
        return {"error": proc.stderr.strip() or "exiftool failed"}
    except FileNotFoundError:
        return {"error": "exiftool not installed"}
    except Exception as e:
        return {"error": f"exiftool error: {e}"}


def run_qpdf_check(file_path: str) -> Dict[str, Any]:
    """Run qpdf --check for PDFs and capture diagnostics."""
    try:
        proc = subprocess.run(
            ["qpdf", "--check", file_path],
            capture_output=True,
            text=True,
            check=False
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result = {
            "return_code": proc.returncode,
            "output": output.strip()
        }
        lowered = output.lower()
        result["has_errors"] = ("error" in lowered)
        result["has_warnings"] = ("warning" in lowered)
        return result
    except FileNotFoundError:
        return {"error": "qpdf not installed"}
    except Exception as e:
        return {"error": f"qpdf error: {e}"}

def ai_tampering_assessment(file_name: str, file_type: str, file_hash: str, metadata: Dict[str, Any], heuristics: List[Dict[str, Any]]):
    client = OpenAI(api_key=st.session_state.openai_api_key)
    system_prompt = (
        "You are a forensic document and image analyst. Given metadata and heuristic findings, "
        "assess likelihood of tampering and recommend next steps."
    )
    user_payload = {
        'file_name': file_name,
        'file_type': file_type,
        'file_hash_sha256': file_hash,
        'metadata': metadata,
        'heuristic_findings': heuristics,
        'desired_output': {
            'tampering_risk_score': '0-100',
            'risk_level': 'low/medium/high',
            'suspicious_fields': ['list of metadata fields of concern'],
            'likely_editing_tools': ['e.g., Photoshop, GIMP, Word, PDF24'],
            'recommended_tests': ['ELA, shadow/lighting analysis, copy-move, signature/XMP checks, hashes'],
            'overall_assessment': 'short paragraph'
        }
    }
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def get_data_folder_documents() -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    data_path = Path("data")
    if data_path.exists():
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]:
                documents.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": file_path.suffix.lower().lstrip('.'),
                    "relative_path": str(file_path.relative_to(data_path))
                })
    return documents


# Sidebar: API keys and data selection
with st.sidebar:
    st.header("API Keys")
    if api_keys_loaded['openai_api_key']:
        st.success("OpenAI API key loaded from .env file")
    else:
        st.warning("OpenAI API key not found in .env file")
    openai_api_key = st.text_input("Override OpenAI API key", type="password")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("OpenAI API key override applied")
    init_language_selector()

    st.markdown("---")
    st.header("Select from data/")
    data_docs = get_data_folder_documents()
    if data_docs:
        selected_doc = st.selectbox(
            "Choose a file from data/ (PDF/JPG/PNG)",
            data_docs,
            format_func=lambda x: f"{x['name']} ({x['relative_path']})"
        )
        if selected_doc:
            st.session_state.selected_document = selected_doc
    else:
        st.info("No supported files found in data/.")


if 'openai_api_key' not in st.session_state:
    st.warning("OpenAI API key is required. Add it in .env or enter it in the sidebar.")

tabs = st.tabs(["Analyze", "Results", "Suggestions"])

with tabs[0]:
    st.subheader("Upload or Use a File from data/")
    uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

    file_source = None
    tmp_path = None
    name = None
    doc_type = None

    if uploaded is not None:
        # Persist upload to a secure temporary file for consistent processing
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        file_source = 'upload'
        name = uploaded.name
        doc_type = Path(uploaded.name).suffix.lower().lstrip('.')
    elif st.session_state.selected_document:
        file_source = 'data'
        tmp_path = st.session_state.selected_document['path']
        name = st.session_state.selected_document['name']
        doc_type = st.session_state.selected_document['type']

    if tmp_path and os.path.exists(tmp_path):
        st.info(f"Selected file: {name}")
        st.write(f"Source: {'Uploaded' if file_source=='upload' else 'data/'}")
        st.code(tmp_path)

        # Extract metadata
        try:
            if doc_type == 'pdf':
                metadata = extract_pdf_metadata(tmp_path)
            else:
                metadata = extract_image_metadata(tmp_path)
        except Exception as e:
            st.error(f"Failed to read metadata: {e}")
            metadata = {}

        # External tools (best-effort)
        exiftool_data = run_exiftool(tmp_path)
        qpdf_result = run_qpdf_check(tmp_path) if doc_type == 'pdf' else {}
        external = {
            'exiftool': exiftool_data,
            'qpdf_check': qpdf_result
        }

        # Heuristics
        heuristics = run_heuristics(doc_type, metadata)
        heuristics.extend(augment_heuristics_with_external(doc_type, exiftool_data, qpdf_result))

        # Compute hash and basic file stats
        file_hash = compute_file_hash(tmp_path)
        stats = os.stat(tmp_path)
        basic = {
            'file_name': name,
            'file_type': doc_type,
            'size_bytes': stats.st_size,
            'sha256': file_hash
        }

        # Show quick view
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Basic Info")
            st.json(basic)
        with c2:
            st.subheader("Heuristic Flags")
            if heuristics:
                for i, h in enumerate(heuristics):
                    badge = {'high': '🔴', 'medium': '🟠', 'low': '🟡'}.get(h.get('severity', 'low').lower(), '🟡')
                    st.write(f"{badge} {h['indicator']} ({h['severity']})")
                    st.caption(h['explanation'])
            else:
                st.success("No heuristic anomalies detected")

        st.subheader("Raw Metadata")
        st.json(metadata)
        with st.expander("External Tool Output (exiftool/qpdf)"):
            st.json(external)

        # Run AI assessment
        if 'openai_api_key' in st.session_state and st.button("Run AI Tampering Analysis"):
            with st.spinner("Analyzing with AI..."):
                try:
                    ai_result = ai_tampering_assessment(name, doc_type, file_hash, metadata, heuristics)
                    st.session_state.tampering_results = {
                        'basic': basic,
                        'metadata': metadata,
                        'heuristics': heuristics,
                        'ai_assessment': ai_result,
                        'external': external
                    }
                    st.success("AI assessment completed")
            
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
    else:
        st.info("Upload a file or select one from the sidebar to begin.")


with tabs[1]:
    st.subheader("Assessment Results")
    if st.session_state.tampering_results:
        res = st.session_state.tampering_results
        # Metrics
        c1, c2 = st.columns(2)
        with c1:
            score = res['ai_assessment'].get('tampering_risk_score', 0)
            st.metric("Tampering Risk", f"{score}/100")
            st.progress(min(max(int(score), 0), 100) / 100)
        with c2:
            level = (res['ai_assessment'].get('risk_level') or 'unknown').upper()
            if level == 'LOW':
                st.success(f"Risk Level: {level}")
            elif level == 'MEDIUM':
                st.warning(f"Risk Level: {level}")
            else:
                st.error(f"Risk Level: {level}")

        st.markdown("### Suspicious Fields")
        for f in res['ai_assessment'].get('suspicious_fields', []) or []:
            st.write(f"- {f}")

        st.markdown("### Likely Editing Tools")
        for t in res['ai_assessment'].get('likely_editing_tools', []) or []:
            st.write(f"- {t}")

        st.markdown("### Recommended Next Tests")
        for r in res['ai_assessment'].get('recommended_tests', []) or []:
            st.write(f"- {r}")

        st.markdown("### Overall Assessment")
        st.write(res['ai_assessment'].get('overall_assessment', 'No assessment available'))

        st.markdown("### Export")
        export_str = json.dumps(res, indent=2)
        st.download_button(
            label="Download Tampering Report (JSON)",
            data=export_str,
            file_name="document_tampering_report.json",
            mime="application/json"
        )
    else:
        st.info("No results yet. Run an analysis in the Analyze tab.")


with tabs[2]:
    st.subheader("Additional Strategies for Tampering Detection")
    st.markdown(
        """
        - Hash comparison: Compare file hashes against the original or known-good versions.
        - PDF internals: Inspect XMP metadata, digital signatures, and incremental update sections.
        - Visual forensics (images):
          - Error Level Analysis (ELA) to spot recompressed regions
          - Copy–move detection to find duplicated patches
          - Lighting/shadow consistency checks
          - JPEG quantization table analysis to identify edit traces
        - OCR consistency: Extract text and compare fonts/kerning; check for rasterized text overlays.
        - Cross-file corroboration: Ensure timestamps, IDs, and amounts match across claim, invoice, and policy.
        - Device provenance: Validate EXIF Make/Model vs. reporter’s device and geolocation plausibility.
        - Chain of custody: Enforce and verify upload logs, signatures, and watermarking.
        - Content duplication: Reverse image search to detect reused images from the web.
        """
    )


