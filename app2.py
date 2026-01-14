import streamlit as st
import base64
import json
import re
import csv
import io
import tempfile
import os

from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# =========================================================
# SUPPORTED VERTEX AI LOCATIONS
# =========================================================
VERTEX_LOCATIONS = [
    "us-central1",
    "us-east1",
    "us-west1",
    "us-west4",
    "europe-west1",
    "europe-west2",
    "europe-west4",
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast3",
    "asia-south1",
    "australia-southeast1"
]

# =========================================================
# SESSION STATE
# =========================================================
if "extracted_json" not in st.session_state:
    st.session_state.extracted_json = None

# =========================================================
# HELPERS
# =========================================================
def get_project_id_from_sa(uploaded_file):
    try:
        sa_json = json.loads(uploaded_file.getvalue().decode("utf-8"))
        return sa_json.get("project_id")
    except Exception:
        return None


def load_prompt_from_file(uploaded_file) -> str:
    return uploaded_file.read().decode("utf-8")


def get_first_response_text_from_stream(stream, status_box):
    final_text = ""
    for chunk in stream:
        if hasattr(chunk, "text") and chunk.text:
            final_text += chunk.text
            status_box.info("🔄 Gemini processing...")
    return final_text


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
        items[parent_key] = obj
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
            if isinstance(v, list) and v and isinstance(v[0], dict)
        }

        if array_fields:
            for array_key, array_items in array_fields.items():
                for item in array_items:
                    row = flat.copy()
                    row.pop(array_key)
                    row.update(flatten_json(item, array_key))
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


def safe_json_loads(text: str):
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        raise ValueError("No JSON found in Gemini response")

    json_str = match.group(1)
    json_str = json_str.replace("\n", " ")
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    return json.loads(json_str)

# =========================================================
# GEMINI EXTRACTION
# =========================================================
def extract_from_pdf(
    pdf_path: str,
    prompt_text: str,
    service_account_path: str,
    project_id: str,
    location: str,
    status_box
) -> dict:

    credentials = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    vertexai.init(
        project=project_id,
        location=location,
        credentials=credentials
    )

    model = GenerativeModel("gemini-2.5-flash")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = Part.from_dict({
        "inline_data": {
            "mime_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode("utf-8")
        }
    })

    status_box.info("📤 Sending PDF to Gemini...")

    response_stream = model.generate_content(
        contents=[pdf_part, prompt_text],
        generation_config=GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        ),
        stream=True
    )

    text = get_first_response_text_from_stream(response_stream, status_box)
    parsed = safe_json_loads(text)

    if isinstance(parsed, list) and len(parsed) == 1:
        return parsed[0]

    return parsed

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Bank Statement Extractor", layout="wide")

st.title("📄 Bank Statement Extraction")
st.caption("Gemini • Vertex AI • Auto Project Detection • JSON → CSV")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔐 Authentication")

    service_account_file = st.file_uploader(
        "Service Account JSON",
        type=["json"]
    )

    project_id = None
    if service_account_file:
        project_id = get_project_id_from_sa(service_account_file)
        if project_id:
            st.text_input(
                "GCP Project ID (auto-detected)",
                value=project_id,
                disabled=True
            )
        else:
            st.error("❌ project_id not found in service account")

    location = st.selectbox(
        "Vertex AI Location",
        VERTEX_LOCATIONS,
        index=VERTEX_LOCATIONS.index("us-central1")
    )

# ---------------- MAIN ----------------
left_col, right_col = st.columns([1, 1.4])

with left_col:
    st.header("📂 Inputs")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    prompt_file = st.file_uploader("Upload Prompt (.txt)", type=["txt"])
    extract_btn = st.button("🚀 Run Extraction", use_container_width=True)
    status_box = st.empty()

with right_col:
    st.header("📊 Extracted JSON")
    json_placeholder = st.empty()
    download_json_placeholder = st.empty()
    convert_csv_placeholder = st.empty()

# =========================================================
# ACTIONS
# =========================================================
if extract_btn:
    if not pdf_file or not prompt_file or not service_account_file:
        st.error("❗ Upload PDF, Prompt, and Service Account JSON")
        st.stop()

    if not project_id:
        st.error("❗ Invalid service account (project_id missing)")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, pdf_file.name)
        sa_path = os.path.join(tmpdir, service_account_file.name)

        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        with open(sa_path, "wb") as f:
            f.write(service_account_file.read())

        prompt_text = load_prompt_from_file(prompt_file)

        status_box.info("⏳ Extracting...")

        try:
            st.session_state.extracted_json = extract_from_pdf(
                pdf_path,
                prompt_text,
                sa_path,
                project_id,
                location,
                status_box
            )
            status_box.success("✅ Extraction completed")
        except Exception as e:
            status_box.error("❌ Extraction failed")
            st.exception(e)

# =========================================================
# RENDER RESULTS
# =========================================================
if st.session_state.extracted_json:
    json_placeholder.json(st.session_state.extracted_json)

    download_json_placeholder.download_button(
        "⬇️ Download JSON",
        data=json.dumps(st.session_state.extracted_json, indent=2),
        file_name="output.json",
        mime="application/json"
    )

    convert_csv_placeholder.download_button(
        "🔄 Convert JSON → CSV",
        data=json_to_csv_string(st.session_state.extracted_json),
        file_name="output.csv",
        mime="text/csv"
    )
