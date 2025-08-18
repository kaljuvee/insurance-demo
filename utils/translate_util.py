import os
import json
from pathlib import Path
from typing import List
from openai import OpenAI
from dotenv import load_dotenv


def get_openai_client() -> OpenAI:
    # Load from .env (python-dotenv)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Ensure OPENAI_API_KEY is set in your .env file."
        )
    return OpenAI(api_key=api_key)


def translate_text_to_estonian(text: str) -> str:
    """Translate any text/markdown to Estonian using OpenAI."""
    client = get_openai_client()
    system_prompt = (
        "You are a professional translator. Translate all user-provided content into Estonian (et). "
        "Preserve formatting (especially Markdown), tone, and intent. Do not add commentary."
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content


def translate_markdown_file_to_estonian(src_path: str, dest_path: str) -> None:
    src = Path(src_path)
    dst = Path(dest_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    translated = translate_text_to_estonian(text)
    dst.write_text(translated, encoding="utf-8")


def batch_translate_markdowns_to_utils_estonian(sources: List[str]) -> None:
    """Batch-translate markdown files and store in utils/estonian/<name>.et.md"""
    base_out = Path("utils/estonian")
    base_out.mkdir(parents=True, exist_ok=True)
    for src in sources:
        src_path = Path(src)
        if not src_path.exists():
            continue
        stem = src_path.name
        out_path = base_out / f"{stem}.et.md"
        translate_markdown_file_to_estonian(str(src_path), str(out_path))


def default_markdown_sources() -> List[str]:
    """Known markdown sources in the repo to translate."""
    sources: List[str] = []
    # Readme and docs
    for p in ["README.md", "docs/if-insurance-ai-demo-user-guide.md"]:
        if Path(p).exists():
            sources.append(p)
    # Prompts
    prompts_dir = Path("prompts")
    if prompts_dir.exists():
        for p in prompts_dir.glob("*.md"):
            sources.append(str(p))
    return sources


def generate_estonian_markdowns() -> None:
    sources = default_markdown_sources()
    batch_translate_markdowns_to_utils_estonian(sources)


