import streamlit as st
import os
import json
import base64
import hashlib
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import PyPDF2
from PIL import Image, ExifTags
from openai import OpenAI
from utils import load_api_keys

st.set_page_config(
    page_title="Batch Processing",
    page_icon="📦",
    layout="wide"
)

st.title("Batch Processing: Data Folder Scan")
st.markdown("Scan all PDFs and images under `data/` for profiling and tampering checks, with optional AI analyses.")

# Load API keys
api_keys_loaded = load_api_keys()


# ---------- Helpers ----------
def compute_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_pdf_date(date_str: str):
    if not date_str:
        return None
    s = date_str
    if s.startswith('D:'):
        s = s[2:]
    try:
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S")
    except Exception:
        try:
            return datetime.strptime(s[:8], "%Y%m%d")
        except Exception:
            return None


def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        info = reader.metadata or {}
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
        exif_data = {}
        try:
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    try:
                        exif_data[str(tag)] = value if isinstance(value, (int, float, str)) else str(value)
                    except Exception:
                        exif_data[str(tag)] = str(value)
        except Exception:
            pass
        meta['exif'] = exif_data
        extra_info = {}
        for k, v in img.info.items():
            if k != 'exif':
                extra_info[k] = v if isinstance(v, (int, float, str)) else str(v)
        if extra_info:
            meta['extra_info'] = extra_info
    return meta


def run_heuristics(document_type: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not metadata:
        findings.append({
            'indicator': 'Metadata missing or empty',
            'severity': 'medium',
            'explanation': 'Document has no accessible metadata; not conclusive but reduces provenance confidence.'
        })

    if document_type == 'pdf':
        creation = parse_pdf_date(metadata.get('CreationDate', ''))
        mod = parse_pdf_date(metadata.get('ModDate', ''))
        if creation and mod and mod < creation:
            findings.append({
                'indicator': 'Modification date precedes creation date',
                'severity': 'high',
                'explanation': 'Inconsistent timestamps can indicate manual metadata editing or tampering.'
            })
        producer = (metadata.get('Producer') or '').lower()
        if any(tool in producer for tool in ['photoshop', 'gimp', 'image editor']):
            findings.append({
                'indicator': 'PDF producer suggests raster editor',
                'severity': 'medium',
                'explanation': 'PDF appears processed by an image editor which may be unusual for invoices/contracts.'
            })
        if not any(k in metadata for k in ['Author', 'Creator', 'Producer', 'Title']):
            findings.append({
                'indicator': 'Minimal descriptive fields',
                'severity': 'low',
                'explanation': 'Missing typical fields (Author/Creator/Title) can be a weak signal of sanitization.'
            })
    else:
        exif = metadata.get('exif', {})
        software = (exif.get('Software') or metadata.get('extra_info', {}).get('Software') or '').lower()
        if any(s in software for s in ['photoshop', 'gimp', 'snapseed', 'lightroom', 'picsart']):
            findings.append({
                'indicator': 'Edited with image manipulation software',
                'severity': 'medium',
                'explanation': 'EXIF Software shows editing tool; inspect for manipulation.'
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
                'explanation': 'Absence of EXIF isn’t conclusive but reduces provenance confidence.'
            })
    return findings


def run_exiftool(file_path: str) -> Dict[str, Any]:
    try:
        proc = subprocess.run(["exiftool", "-json", "-n", file_path], capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            data = json.loads(proc.stdout)
            return data[0] if isinstance(data, list) and data else {}
        return {"error": proc.stderr.strip() or "exiftool failed"}
    except FileNotFoundError:
        return {"error": "exiftool not installed"}
    except Exception as e:
        return {"error": f"exiftool error: {e}"}


def run_qpdf_check(file_path: str) -> Dict[str, Any]:
    try:
        proc = subprocess.run(["qpdf", "--check", file_path], capture_output=True, text=True, check=False)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result = {"return_code": proc.returncode, "output": output.strip()}
        lowered = output.lower()
        result["has_errors"] = ("error" in lowered)
        result["has_warnings"] = ("warning" in lowered)
        return result
    except FileNotFoundError:
        return {"error": "qpdf not installed"}
    except Exception as e:
        return {"error": f"qpdf error: {e}"}


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
                'explanation': 'EXIF Software shows common image editors; inspect for manipulation.'
            })
    return extra


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


def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def ai_image_fraud_detection(image_base64: str) -> Dict[str, Any]:
    client = OpenAI(api_key=st.session_state.openai_api_key)
    prompt = """
    Analyze this image for potential fraud indicators (digital manipulation, staged damage, inconsistent lighting/shadows).
    Return JSON with: fraud_risk_score (0-100), risk_level, fraud_indicators [indicator, severity, confidence, explanation],
    authenticity_assessment, recommendations.
    """
    messages = [
        {"role": "system", "content": "You are an expert image forensics analyst."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def ai_image_false_pos_neg(image_base64: str) -> Dict[str, Any]:
    client = OpenAI(api_key=st.session_state.openai_api_key)
    prompt = """
    Assess this image for potential false positives/negatives in damage claims. Return JSON with:
    false_positive_score, false_negative_score, overall_accuracy_score, assessment_confidence,
    false_positive_indicators [indicator, severity, explanation], false_negative_indicators [...],
    context_analysis, recommendations.
    """
    messages = [
        {"role": "system", "content": "You are an expert damage assessment auditor."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def list_data_files() -> List[Dict[str, Any]]:
    data_path = Path("data")
    files: List[Dict[str, Any]] = []
    if data_path.exists():
        for p in data_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]:
                files.append({
                    'name': p.name,
                    'path': str(p),
                    'type': p.suffix.lower().lstrip('.'),
                    'relative_path': str(p.relative_to(data_path))
                })
    return files


# ---------- Sidebar ----------
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

    st.markdown("---")
    st.header("Batch Options")
    run_ai_tampering = st.checkbox("Run AI tampering assessment (PDF & images)", value=False)
    run_ai_image_fraud = st.checkbox("Run AI image fraud detection", value=False)
    run_ai_false_pos_neg = st.checkbox("Run AI false positive/negative detection", value=False)

    st.caption("Note: AI options may incur token costs and take longer.")


if 'openai_api_key' not in st.session_state and (run_ai_tampering or run_ai_image_fraud or run_ai_false_pos_neg):
    st.warning("OpenAI API key is required for AI analyses. Add it in .env or enter it in the sidebar.")


# ---------- Main ----------
files = list_data_files()
st.write(f"Found {len(files)} supported files in data/ (PDF/JPG/PNG)")

if st.button("Run Batch Scan"):
    if len(files) == 0:
        st.warning("No files found to scan.")
    else:
        results: List[Dict[str, Any]] = []
        progress_bar = st.progress(0)
        status = st.empty()
        for idx, f in enumerate(files, start=1):
            status.write(f"Processing {idx}/{len(files)}: {f['relative_path']}")
            try:
                file_path = f['path']
                doc_type = f['type']
                # Basic info
                file_hash = compute_file_hash(file_path)
                stats = os.stat(file_path)
                basic = {
                    'file_name': f['name'],
                    'file_type': doc_type,
                    'relative_path': f['relative_path'],
                    'size_bytes': stats.st_size,
                    'sha256': file_hash
                }

                # Metadata
                try:
                    if doc_type == 'pdf':
                        metadata = extract_pdf_metadata(file_path)
                    else:
                        metadata = extract_image_metadata(file_path)
                except Exception as e:
                    metadata = {"error": f"metadata error: {e}"}

                # External tools
                exiftool_data = run_exiftool(file_path)
                qpdf_result = run_qpdf_check(file_path) if doc_type == 'pdf' else {}

                # Heuristics
                heuristics = run_heuristics(doc_type, metadata if isinstance(metadata, dict) else {})
                heuristics.extend(augment_heuristics_with_external(doc_type, exiftool_data, qpdf_result))

                # AI assessments (optional)
                ai_tamper = None
                ai_img_fraud = None
                ai_img_fpfn = None
                if run_ai_tampering and 'openai_api_key' in st.session_state:
                    try:
                        ai_tamper = ai_tampering_assessment(basic['file_name'], doc_type, file_hash, metadata, heuristics)
                    except Exception as e:
                        ai_tamper = {"error": f"ai tampering error: {e}"}
                if doc_type != 'pdf' and (run_ai_image_fraud or run_ai_false_pos_neg) and 'openai_api_key' in st.session_state:
                    try:
                        img_b64 = encode_image_to_base64(file_path)
                        if run_ai_image_fraud:
                            try:
                                ai_img_fraud = ai_image_fraud_detection(img_b64)
                            except Exception as e:
                                ai_img_fraud = {"error": f"ai image fraud error: {e}"}
                        if run_ai_false_pos_neg:
                            try:
                                ai_img_fpfn = ai_image_false_pos_neg(img_b64)
                            except Exception as e:
                                ai_img_fpfn = {"error": f"ai fp/fn error: {e}"}
                    except Exception as e:
                        pass

                results.append({
                    'basic': basic,
                    'metadata': metadata,
                    'exiftool': exiftool_data,
                    'qpdf': qpdf_result,
                    'heuristics': heuristics,
                    'ai_tampering': ai_tamper,
                    'ai_image_fraud': ai_img_fraud,
                    'ai_false_pos_neg': ai_img_fpfn
                })
            except Exception as e:
                results.append({
                    'basic': {'file_name': f['name'], 'relative_path': f['relative_path'], 'file_type': f['type']},
                    'error': str(e)
                })
            progress_bar.progress(idx / len(files))

        st.success("Batch scan completed")

        # Summary table
        def summarize(rec: Dict[str, Any]) -> Dict[str, Any]:
            basic = rec.get('basic', {})
            tamper_score = None
            level = None
            if isinstance(rec.get('ai_tampering'), dict):
                tamper_score = rec['ai_tampering'].get('tampering_risk_score')
                level = rec['ai_tampering'].get('risk_level')
            qpdf_flags = None
            if isinstance(rec.get('qpdf'), dict):
                if rec['qpdf'].get('has_errors'):
                    qpdf_flags = 'errors'
                elif rec['qpdf'].get('has_warnings'):
                    qpdf_flags = 'warnings'
            exif_sw = None
            exiftool_data = rec.get('exiftool')
            if isinstance(exiftool_data, dict):
                exif_sw = exiftool_data.get('Software')
            return {
                'path': basic.get('relative_path'),
                'type': basic.get('file_type'),
                'size_bytes': basic.get('size_bytes'),
                'tampering_score': tamper_score,
                'tampering_level': level,
                'qpdf': qpdf_flags,
                'exif_software': exif_sw,
                'heuristics_count': len(rec.get('heuristics') or [])
            }

        summary_rows = [summarize(r) for r in results]
        df = pd.DataFrame(summary_rows)
        st.subheader("Summary")
        st.dataframe(df, use_container_width=True)

        # Detailed view
        st.subheader("Details")
        for i, rec in enumerate(results):
            basic = rec.get('basic', {})
            with st.expander(f"{i+1}. {basic.get('relative_path', basic.get('file_name', 'file'))}"):
                st.json(rec)

        # Exports
        st.subheader("Export Results")
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=json_data,
            file_name="batch_scan_report.json",
            mime="application/json"
        )
        st.download_button(
            label="Download CSV Summary",
            data=df.to_csv(index=False),
            file_name="batch_scan_summary.csv",
            mime="text/csv"
        )


