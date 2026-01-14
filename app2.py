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
    "us-central1", "us-east1", "us-west1",
    "us-west4", "europe-west1", "asia-south1"
]
MAX_CONCURRENCY = 4

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
def transactions_to_csv(data: dict) -> str:
    """
    Convert extracted bank statement JSON to CSV
    with ONE ROW PER TRANSACTION.
    """

    if not isinstance(data, dict):
        return ""

    transactions = data.get("transactions", [])
    if not transactions:
        return ""

    output = io.StringIO()

    # Base (repeated) fields
    base_fields = {
        "bank_name": data.get("bank_name"),
        "account_holder_name": data.get("account_holder_name"),
        "account_number": data.get("account_number"),
        "currency": data.get("currency"),
        "statement_from": (data.get("statement_period") or {}).get("from"),
        "statement_to": (data.get("statement_period") or {}).get("to"),
    }

    rows = []
    for txn in transactions:
        row = base_fields.copy()
        row.update({
            "txn_date": txn.get("date"),
            "description": txn.get("description"),
            "amount": txn.get("amount"),
            "balance": txn.get("balance"),
            "type": txn.get("type"),
            "category": txn.get("category"),
        })
        rows.append(row)

    fieldnames = [
        "bank_name",
        "account_holder_name",
        "account_number",
        "currency",
        "statement_from",
        "statement_to",
        "txn_date",
        "description",
        "amount",
        "balance",
        "type",
        "category",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()

def safe_json_loads(text: str):
    match = re.search(r"(\{[\s\S]*$|\[[\s\S]*$)", text)
    if not match:
        raise ValueError("No JSON found in Gemini response")

    json_str = re.sub(r"```json|```", "", match.group(1))
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

    raise ValueError("Unrecoverable malformed JSON")


# =========================================================
# JSON → CSV (SCHEMA-AGNOSTIC)
# =========================================================
def flatten_json(obj, parent_key="", sep="."):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep))
    elif isinstance(obj, list):
        items[parent_key] = json.dumps(obj)
    else:
        items[parent_key] = obj
    return items


def json_to_csv_string(data) -> str:
    output = io.StringIO()
    rows = []

    if isinstance(data, list):
        for item in data:
            rows.append(flatten_json(item))

    elif isinstance(data, dict):
        flat = flatten_json(data)
        array_fields = {
            k: v for k, v in flat.items()
            if isinstance(v, list)
        }

        if array_fields:
            for k, v in array_fields.items():
                for item in v:
                    row = flat.copy()
                    row.pop(k)
                    row.update(flatten_json(item, k))
                    rows.append(row)
        else:
            rows.append(flat)

    if not rows:
        return ""

    fieldnames = sorted({k for row in rows for k in row.keys()})
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()

# =========================================================
# PDF PAGE SPLIT
# =========================================================
def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        single = fitz.open()
        single.insert_pdf(doc, from_page=i, to_page=i)
        pages.append((i, single.write()))
    return pages


def merge_page_results(results):
    merged_dict = {}
    merged_lists = []

    for _, page_data in sorted(results, key=lambda x: x[0]):
        if isinstance(page_data, list):
            merged_lists.extend(page_data)
        elif isinstance(page_data, dict):
            for k, v in page_data.items():
                if isinstance(v, list):
                    merged_dict.setdefault(k, []).extend(v)
                else:
                    if k not in merged_dict or merged_dict[k] in (None, "", {}):
                        merged_dict[k] = v

    if merged_dict == {} and merged_lists:
        return merged_lists

    if merged_lists:
        merged_dict.setdefault("items", []).extend(merged_lists)

    return merged_dict

# =========================================================
# GEMINI STREAMING (SYNC)
# =========================================================
def call_gemini_stream_sync(pdf_bytes, prompt, model, page_index, progress_queue):
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
            progress_queue.put_nowait(page_index)

    return page_index, safe_json_loads(text)

# =========================================================
# ASYNC WRAPPER
# =========================================================
async def call_gemini_stream_async(pdf_bytes, prompt, model, semaphore, page_index, progress_queue):
    async with semaphore:
        return await asyncio.to_thread(
            call_gemini_stream_sync,
            pdf_bytes,
            prompt,
            model,
            page_index,
            progress_queue
        )

# =========================================================
# ASYNC EXTRACTION
# =========================================================
async def extract_parallel_pages_streaming_async(
    pdf_path, prompt, creds, project_id, location, progress_bar, status_box
):
    vertexai.init(project=project_id, location=location, credentials=creds)
    model = GenerativeModel("gemini-2.5-flash")

    pages = extract_pages(pdf_path)
    total_pages = len(pages)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    progress_queue = asyncio.Queue()
    completed_pages = set()

    async def progress_watcher():
        while len(completed_pages) < total_pages:
            page_idx = await progress_queue.get()
            completed_pages.add(page_idx)
            progress_bar.progress(len(completed_pages) / total_pages)
            status_box.info(f"📄 Processing page {page_idx + 1}/{total_pages}")

    watcher = asyncio.create_task(progress_watcher())

    tasks = [
        call_gemini_stream_async(b, prompt, model, semaphore, i, progress_queue)
        for i, b in pages
    ]

    results = await asyncio.gather(*tasks)
    await watcher

    return merge_page_results(results)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Async Gemini Bank Statement Extractor", layout="wide")
st.title("📄 Async Gemini Bank Statement Extractor")

with st.sidebar:
    service_account_file = st.file_uploader("Service Account JSON", type=["json"])
    project_id = get_project_id_from_sa(service_account_file) if service_account_file else None
    if project_id:
        st.text_input("Project ID", project_id, disabled=True)

    location = st.selectbox("Vertex AI Location", VERTEX_LOCATIONS)

left, right = st.columns([1, 1.4])

with left:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    prompt_file = st.file_uploader("Upload Prompt (.txt)", type=["txt"])
    run = st.button("🚀 Run Extraction", use_container_width=True)
    status_box = st.empty()
    progress_bar = st.progress(0.0)

with right:
    json_out = st.empty()
    download_json = st.empty()
    download_csv = st.empty()

# =========================================================
# ACTION
# =========================================================
if run:
    if not all([pdf_file, prompt_file, service_account_file]):
        st.error("Upload PDF, Prompt, and Service Account JSON")
        st.stop()

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

        try:
            st.session_state.extracted_json = asyncio.run(
                extract_parallel_pages_streaming_async(
                    pdf_path,
                    prompt,
                    creds,
                    project_id,
                    location,
                    progress_bar,
                    status_box
                )
            )
            status_box.success("✅ Extraction completed")
        except Exception as e:
            status_box.error("❌ Extraction failed")
            st.exception(e)

# =========================================================
# RENDER + DOWNLOADS
# =========================================================
if st.session_state.extracted_json:
    json_out.json(st.session_state.extracted_json)

    download_json.download_button(
        "⬇️ Download JSON",
        json.dumps(st.session_state.extracted_json, indent=2),
        "output.json",
        mime="application/json"
    )

    download_csv.download_button(
    "⬇️ Download Transactions CSV",
    transactions_to_csv(st.session_state.extracted_json),
    "transactions.csv",
    mime="text/csv"
)

