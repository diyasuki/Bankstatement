import streamlit as st
import base64
import json
import re
import csv
import io
import tempfile
import os
import fitz  # PyMuPDF
import asyncio

from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# =========================================================
# CONFIG
# =========================================================
VERTEX_LOCATIONS = [
    "us-central1", "us-east1", "us-west1", "us-west4",
    "europe-west1", "asia-east1", "asia-south1"
]

MAX_CONCURRENCY = 4  # safe for Gemini

# =========================================================
# SESSION STATE
# =========================================================
if "extracted_json" not in st.session_state:
    st.session_state.extracted_json = None

# =========================================================
# HELPERS
# =========================================================
def get_project_id_from_sa(uploaded_file):
    sa = json.loads(uploaded_file.getvalue().decode("utf-8"))
    return sa.get("project_id")


def safe_json_loads(text: str):
    match = re.search(r"(\{[\s\S]*$|\[[\s\S]*$)", text)
    if not match:
        raise ValueError("No JSON found")

    json_str = match.group(1)
    json_str = re.sub(r"```json|```", "", json_str)
    json_str = json_str.replace("\n", " ")
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    for i in range(len(json_str), 0, -1):
        try:
            candidate = json_str[:i]
            if candidate.count("{") == candidate.count("}"):
                return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("Unrecoverable JSON")


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page_pdf = fitz.open()
        page_pdf.insert_pdf(doc, from_page=i, to_page=i)
        pages.append((i, page_pdf.write()))
    return pages


def merge_page_results(results):
    merged = {}
    arrays = {}

    for _, page_data in sorted(results, key=lambda x: x[0]):
        for k, v in page_data.items():
            if isinstance(v, list):
                arrays.setdefault(k, []).extend(v)
            elif k not in merged or merged[k] in (None, "", {}):
                merged[k] = v

    merged.update(arrays)
    return merged

# =========================================================
# GEMINI STREAMING (SYNC)
# =========================================================
def call_gemini_stream_sync(pdf_bytes, prompt, model, page_index, progress_cb=None):
    part = Part.from_dict({
        "inline_data": {
            "mime_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode()
        }
    })

    stream = model.generate_content(
        contents=[part, prompt],
        generation_config=GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        ),
        stream=True
    )

    text = ""
    for chunk in stream:
        if hasattr(chunk, "text") and chunk.text:
            text += chunk.text
            if progress_cb:
                progress_cb(page_index)

    parsed = safe_json_loads(text)
    return page_index, parsed

# =========================================================
# GEMINI STREAMING (ASYNC WRAPPER)
# =========================================================
async def call_gemini_stream_async(
    pdf_bytes,
    prompt,
    model,
    semaphore,
    page_index,
    progress_cb
):
    async with semaphore:
        return await asyncio.to_thread(
            call_gemini_stream_sync,
            pdf_bytes,
            prompt,
            model,
            page_index,
            progress_cb
        )

# =========================================================
# EXTRACTION MODES
# =========================================================
def extract_full_document(pdf_path, prompt, creds, project_id, location):
    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel("gemini-2.5-flash")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    _, result = call_gemini_stream_sync(pdf_bytes, prompt, model, 0)
    return result


async def extract_parallel_pages_streaming_async(
    pdf_path,
    prompt,
    creds,
    project_id,
    location,
    progress_cb
):
    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel("gemini-2.5-flash")

    pages = extract_pages(pdf_path)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        call_gemini_stream_async(
            pdf_bytes=b,
            prompt=prompt,
            model=model,
            semaphore=semaphore,
            page_index=i,
            progress_cb=progress_cb
        )
        for i, b in pages
    ]

    results = await asyncio.gather(*tasks)
    return merge_page_results(results)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Async Gemini PDF Extractor", layout="wide")
st.title("📄 Async Gemini PDF Extraction")
st.caption("Async • Streaming • Page-safe • Ordered merge")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    service_account_file = st.file_uploader("Service Account JSON", type=["json"])

    project_id = get_project_id_from_sa(service_account_file) if service_account_file else None
    if project_id:
        st.text_input("Project ID", project_id, disabled=True)

    location = st.selectbox("Vertex AI Location", VERTEX_LOCATIONS)

    mode = st.radio(
        "Extraction Mode",
        ["Full Document (Single Call)", "Parallel Pages (Async Streaming)"],
        index=1
    )

# ---------------- MAIN ----------------
left, right = st.columns([1, 1.4])

with left:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    prompt_file = st.file_uploader("Upload Prompt (.txt)", type=["txt"])
    run = st.button("🚀 Run Extraction", use_container_width=True)
    status = st.empty()
    progress_bar = st.progress(0.0)

with right:
    output = st.empty()
    dl_json = st.empty()

# =========================================================
# ACTION
# =========================================================
if run:
    if not all([pdf_file, prompt_file, service_account_file]):
        st.error("Upload PDF, Prompt, and Service Account JSON")
        st.stop()

    # Write service account to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as sa_tmp:
        sa_tmp.write(service_account_file.getvalue())
        sa_path = sa_tmp.name

    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        prompt = prompt_file.read().decode()
        total_pages = len(fitz.open(pdf_path))

        completed_pages = set()

        def on_progress(page_idx):
            completed_pages.add(page_idx)
            progress_bar.progress(len(completed_pages) / total_pages)
            status.info(f"📄 Processing page {page_idx + 1}/{total_pages}")

        try:
            if "Parallel" in mode:
                st.session_state.extracted_json = asyncio.run(
                    extract_parallel_pages_streaming_async(
                        pdf_path,
                        prompt,
                        creds,
                        project_id,
                        location,
                        on_progress
                    )
                )
            else:
                st.session_state.extracted_json = extract_full_document(
                    pdf_path,
                    prompt,
                    creds,
                    project_id,
                    location
                )

            status.success("✅ Extraction completed")

        except Exception as e:
            status.error("❌ Extraction failed")
            st.exception(e)

# =========================================================
# RENDER
# =========================================================
if st.session_state.extracted_json:
    output.json(st.session_state.extracted_json)

    dl_json.download_button(
        "⬇ Download JSON",
        json.dumps(st.session_state.extracted_json, indent=2),
        "output.json",
        mime="application/json"
    )
