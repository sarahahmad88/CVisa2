"""
Viza Pilot — AI Visa & Travel Documentation Co-Pilot
-----------------------------------------------------
A hybrid RAG (retrieval-augmented generation) application that helps
Pakistani passport holders prepare Student and Tourist visa applications.

Tech stack:
- Streamlit  : user interface
- PyMuPDF    : native PDF text extraction and PDF page rendering
- Tesseract   : OCR for scanned PDFs and photographed documents
- Sentence-Transformers : embeddings for semantic search
- FAISS      : vector similarity search over the official knowledge base
- Groq       : LLM generation (personalized checklists, extraction, guidance)

Design principle: verified official information (from the bundled country
guides) is never overwritten by the language model. The model may explain,
summarize and phrase things naturally, but every specific rule shown to the
user is grounded in retrieved source text, or clearly marked as unverified
general guidance.
"""

import os
import re
import io
import json
import uuid
import shutil
import threading
from datetime import date, datetime

import numpy as np
import streamlit as st
import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image, ImageOps
except Exception:
    pytesseract = None
    Image = None
    ImageOps = None

# ----------------------------------------------------------------------
# Optional heavy imports are wrapped so the app can still show a friendly
# error instead of crashing outright if a dependency fails to load.
# ----------------------------------------------------------------------
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from groq import Groq
except Exception:
    Groq = None


# ========================================================================
# APP CONFIG
# ========================================================================
st.set_page_config(
    page_title="Viza Pilot",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
OCR_DPI = 220
OCR_MIN_NATIVE_CHARS = 40
OCR_LANG = os.getenv("TESSERACT_LANG", "eng")

# Optional override for local Windows installs, e.g.
# TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
if pytesseract is not None:
    _tesseract_cmd = os.getenv("TESSERACT_CMD")
    if _tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

DISCLAIMER = (
    "Viza Pilot organizes and checks your visa documentation using official "
    "sources where available. It does not make visa decisions and cannot "
    "guarantee approval. Always confirm current requirements with the "
    "relevant embassy, consulate, or visa application centre."
)

# Countries covered by the product concept. Only countries with a file in
# knowledge/ have verified, source-grounded content; others show a clearly
# labeled general checklist until an official guide is added.
COUNTRIES = {
    "Germany":         {"guide_file": "Germany_Visa_Guide_Pakistan", "schengen": True},
    "France":          {"guide_file": "France_Visa_Guide_Pakistan",  "schengen": True},
    "Italy":           {"guide_file": "Italy_Visa_Guide_Pakistan",   "schengen": True},
    "Spain":           {"guide_file": None, "schengen": True},
    "Greece":          {"guide_file": None, "schengen": True},
    "Switzerland":     {"guide_file": None, "schengen": True},
    "Norway":          {"guide_file": None, "schengen": True},
    "United Kingdom":  {"guide_file": None, "schengen": False},
    "Turkey":          {"guide_file": None, "schengen": False},
}

VISA_TYPES = ["Student Visa", "Tourist / Short-Stay Visa"]

# Baseline checklist used for every country/visa combination. Country guides
# add verified detail on top of this; countries without a guide still get
# this general-purpose structure with a clear "not yet verified" notice.
BASELINE_CHECKLIST = [
    {"id": "passport", "category": "Identity", "label": "Valid passport (blank pages, valid well beyond your stay)", "mandatory": True, "visa": "All"},
    {"id": "photos", "category": "Identity", "label": "Recent passport-size photographs", "mandatory": True, "visa": "All"},
    {"id": "cnic", "category": "Identity", "label": "CNIC copy", "mandatory": True, "visa": "All"},
    {"id": "form", "category": "Application", "label": "Completed and signed visa application form", "mandatory": True, "visa": "All"},
    {"id": "purpose_student", "category": "Purpose of Travel", "label": "Admission / enrolment letter from your institution", "mandatory": True, "visa": "Student Visa"},
    {"id": "purpose_tourist", "category": "Purpose of Travel", "label": "Travel itinerary and/or invitation letter", "mandatory": True, "visa": "Tourist / Short-Stay Visa"},
    {"id": "flights", "category": "Purpose of Travel", "label": "Flight reservation (return ticket)", "mandatory": True, "visa": "Tourist / Short-Stay Visa"},
    {"id": "financial", "category": "Financial Evidence", "label": "Bank statements / blocked account / sponsor letter", "mandatory": True, "visa": "All"},
    {"id": "salary_slips", "category": "Financial Evidence", "label": "Recent salary slips or proof of income", "mandatory": False, "visa": "All"},
    {"id": "accommodation", "category": "Accommodation", "label": "Proof of accommodation for the full stay", "mandatory": True, "visa": "All"},
    {"id": "insurance", "category": "Insurance", "label": "Travel / health insurance covering the visa's home region", "mandatory": True, "visa": "All"},
    {"id": "employer_letter", "category": "Purpose of Travel", "label": "Employer or university NOC / confirmation letter", "mandatory": False, "visa": "Tourist / Short-Stay Visa"},
    {"id": "degree_certs", "category": "Academic", "label": "Previous degree certificates", "mandatory": True, "visa": "Student Visa"},
    {"id": "transcripts", "category": "Academic", "label": "Academic transcripts", "mandatory": True, "visa": "Student Visa"},
    {"id": "tuition_proof", "category": "Academic", "label": "Proof of tuition fee payment (if applicable)", "mandatory": False, "visa": "Student Visa"},
]

RAG_QUERIES = [
    "mandatory identity and passport requirements",
    "financial evidence and minimum funds requirement amount",
    "travel or health insurance minimum coverage requirement",
    "proof of accommodation requirement",
    "visa fee amount and service charges",
    "processing time for the application",
    "student visa specific academic and enrolment documents required",
]


# ========================================================================
# USAGE STATS: visits, AI token usage, and satisfaction rating
# ------------------------------------------------------------------------
# Lightweight, file-based counters shared across every visitor's session
# for as long as the app instance stays up. This keeps the MVP simple and
# needs no external database. On Streamlit Community Cloud the file lives
# on the app's own disk, so counts persist across sessions but reset if
# the app is redeployed or goes to sleep and wakes up fresh. For durable,
# long-term analytics, swap load_stats()/save_stats() for a small external
# store (e.g. a Google Sheet or a hosted database) later.
# ========================================================================
STATS_PATH = os.path.join(os.path.dirname(__file__), "data", "usage_stats.json")
_stats_lock = threading.Lock()


def _default_stats():
    return {"visits": 0, "total_tokens": 0, "rating_sum": 0, "rating_count": 0}


def load_stats():
    with _stats_lock:
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            defaults = _default_stats()
            defaults.update(data)
            return defaults
        except Exception:
            return _default_stats()


def save_stats(stats):
    with _stats_lock:
        try:
            os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
            with open(STATS_PATH, "w", encoding="utf-8") as f:
                json.dump(stats, f)
        except Exception:
            pass  # Stats are a nice-to-have; never break the app over them.


def increment_visits():
    stats = load_stats()
    stats["visits"] += 1
    save_stats(stats)
    return stats["visits"]


def add_tokens(count: int):
    if not count:
        return
    stats = load_stats()
    stats["total_tokens"] += int(count)
    save_stats(stats)


def add_rating(score: int):
    stats = load_stats()
    stats["rating_sum"] += int(score)
    stats["rating_count"] += 1
    save_stats(stats)


# ========================================================================
# STYLE — keep the interface clean and free of technical jargon
# ========================================================================
def inject_style():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {padding-top: 2rem; max-width: 1100px;}

        section[data-testid="stSidebar"] {
            background-color: #10141c;
            min-width: 280px !important;
            width: 280px !important;
            max-width: 280px !important;
            transform: translateX(0) !important;
            visibility: visible !important;
            display: block !important;
        }
        /* Keep the left navigation permanently open. */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: 0 !important;
            transform: translateX(0) !important;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {
            display: none !important;
        }
        section[data-testid="stSidebar"] * {
            color: #e8ebf0 !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.15rem;
        }
        section[data-testid="stSidebar"] label {
            padding: 0.5rem 0.6rem;
            border-radius: 8px;
        }

        .vp-card {
            background: #ffffff;
            border: 1px solid #eaecf0;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }
        .vp-badge-ok {
            background:#ecfdf3; color:#087443; padding:2px 10px;
            border-radius:999px; font-size:0.78rem; font-weight:600;
        }
        .vp-badge-warn {
            background:#fff7e6; color:#9a5b00; padding:2px 10px;
            border-radius:999px; font-size:0.78rem; font-weight:600;
        }
        .vp-badge-risk {
            background:#fef3f2; color:#b42318; padding:2px 10px;
            border-radius:999px; font-size:0.78rem; font-weight:600;
        }
        .vp-muted {color:#667085; font-size:0.9rem;}
        .vp-score {font-size:3rem; font-weight:800; margin:0;}
        .vp-footer-disclaimer {
            color:#98a2b3; font-size:0.78rem; margin-top:2rem;
            border-top:1px solid #eaecf0; padding-top:0.75rem;
        }

        /* Floating feedback widget, bottom-right corner */
        div[class*="st-key-vp_feedback_widget"] {
            position: fixed;
            bottom: 18px;
            right: 18px;
            z-index: 9999;
            background: #ffffff;
            border: 1px solid #eaecf0;
            border-radius: 14px;
            padding: 0.7rem 1rem 0.4rem 1rem;
            box-shadow: 0 6px 18px rgba(16,24,40,0.14);
            width: 230px;
        }
        div[class*="st-key-vp_feedback_widget"] .vp-stats-line {
            font-size: 0.7rem;
            color: #98a2b3;
            margin-bottom: 0.35rem;
            line-height: 1.4;
        }
        div[class*="st-key-vp_feedback_widget"] .vp-rate-label {
            font-size: 0.78rem;
            color: #475467;
            font-weight: 600;
            margin-bottom: 0.1rem;
        }
        @media (max-width: 640px) {
            div[class*="st-key-vp_feedback_widget"] { width: 190px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ========================================================================
# KNOWLEDGE BASE (RAG): load guides, chunk, embed, index with FAISS
# ========================================================================
def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tesseract_available() -> bool:
    """Return True when both pytesseract and the Tesseract binary are usable."""
    if pytesseract is None or Image is None:
        return False
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if os.path.isabs(cmd):
        return os.path.exists(cmd)
    return shutil.which(cmd) is not None


def _prepare_image_for_ocr(image):
    """Normalize rotation/contrast so phone photos OCR more reliably."""
    if ImageOps is None:
        return image
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


def _ocr_pil_image(image) -> str:
    if not _tesseract_available():
        return ""
    prepared = _prepare_image_for_ocr(image)
    return pytesseract.image_to_string(prepared, lang=OCR_LANG, config="--oem 3 --psm 6").strip()


def _ocr_pdf_page(page) -> str:
    """Render one PDF page to an image and OCR it with Tesseract."""
    if not _tesseract_available():
        return ""
    zoom = OCR_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return _ocr_pil_image(image)
    finally:
        image.close()


def _extract_text_from_pdf(pdf) -> tuple[str, int]:
    """Use native PDF text first, then OCR only pages that appear scanned."""
    page_texts = []
    ocr_pages = 0
    for page in pdf:
        native = page.get_text("text") or ""
        # A page with only a few stray characters is usually an image-only scan.
        if len(_clean_text(native)) < OCR_MIN_NATIVE_CHARS:
            ocr_text = _ocr_pdf_page(page)
            if ocr_text:
                native = ocr_text
                ocr_pages += 1
        page_texts.append(native.strip())
    return "\n\n".join(t for t in page_texts if t), ocr_pages


def _extract_text_from_file(path: str) -> str:
    """Read .txt or PDF guides; scanned PDF pages fall back to Tesseract OCR."""
    if path.lower().endswith(".pdf"):
        doc = fitz.open(path)
        try:
            text, _ = _extract_text_from_pdf(doc)
            return text
        finally:
            doc.close()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_uploaded_text(uploaded) -> tuple[str, str, str | None]:
    """Read a Streamlit upload and return (text, method, error_message)."""
    file_bytes = uploaded.getvalue()
    name = (uploaded.name or "").lower()
    mime = (uploaded.type or "").lower()

    try:
        if mime == "application/pdf" or name.endswith(".pdf"):
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                text, ocr_pages = _extract_text_from_pdf(pdf)
            finally:
                pdf.close()
            method = "OCR + PDF text" if ocr_pages else "PDF text"
            return text, method, None

        if mime == "text/plain" or name.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore"), "Text", None

        if mime.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg")):
            if not _tesseract_available():
                return "", "OCR unavailable", (
                    "Image reading is not available because Tesseract OCR is not installed on the server."
                )
            image = Image.open(io.BytesIO(file_bytes))
            try:
                text = _ocr_pil_image(image)
            finally:
                image.close()
            return text, "OCR", None

        return "", "Unsupported", "This file type cannot be read automatically."
    except Exception as exc:
        return "", "Read failed", f"Automatic reading failed: {exc}"


def _chunk_text(text: str, chunk_size: int = 850, overlap: int = 150):
    text = _clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 40]


@st.cache_resource(show_spinner="Preparing Viza Pilot's country knowledge base...")
def load_knowledge_base():
    """Builds one FAISS index per country that has a bundled official guide."""
    if SentenceTransformer is None or faiss is None:
        return None, {}

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    kb = {}

    for country, info in COUNTRIES.items():
        guide_file = info.get("guide_file")
        if not guide_file:
            continue

        candidate_paths = [
            os.path.join(KNOWLEDGE_DIR, guide_file + ".pdf"),
            os.path.join(KNOWLEDGE_DIR, guide_file + ".txt"),
        ]
        path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if path is None:
            continue

        raw_text = _extract_text_from_file(path)
        chunks = _chunk_text(raw_text)
        if not chunks:
            continue

        embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        kb[country] = {
            "index": index,
            "chunks": chunks,
            "source_name": os.path.basename(path),
        }

    return model, kb


def retrieve(country: str, query: str, model, kb, k: int = 3):
    if model is None or country not in kb:
        return []
    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype="float32")
    entry = kb[country]
    k = min(k, len(entry["chunks"]))
    if k == 0:
        return []
    scores, idxs = entry["index"].search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({"text": entry["chunks"][idx], "score": float(score), "source": entry["source_name"]})
    return results


def gather_country_context(country: str, model, kb, max_chars: int = 4500):
    """Runs a handful of topical queries and stitches together deduplicated context."""
    seen = set()
    pieces = []
    total_len = 0
    for q in RAG_QUERIES:
        for hit in retrieve(country, q, model, kb, k=2):
            key = hit["text"][:80]
            if key in seen:
                continue
            seen.add(key)
            pieces.append(hit["text"])
            total_len += len(hit["text"])
            if total_len >= max_chars:
                return "\n---\n".join(pieces)
    return "\n---\n".join(pieces)


# ========================================================================
# GROQ (LLM) LAYER
# ========================================================================
def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key or Groq is None:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def ask_groq(system_prompt: str, user_prompt: str, max_tokens: int = 900, temperature: float = 0.2):
    client = get_groq_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            usage = getattr(resp, "usage", None)
            total = getattr(usage, "total_tokens", None) if usage else None
            if total:
                add_tokens(total)
                st.session_state["session_tokens"] = st.session_state.get("session_tokens", 0) + total
        except Exception:
            pass
        return resp.choices[0].message.content
    except Exception:
        return None


def _extract_json_block(text: str):
    if not text:
        return None
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


# ---- Personalized checklist enrichment -------------------------------
@st.cache_data(show_spinner=False)
def generate_verified_notes(country: str, visa_type: str, context: str):
    """
    Asks the model to summarize verified, source-grounded detail notes for
    the checklist (fees, thresholds, insurance minimums, procedure notes).
    The model is instructed to use ONLY the supplied context.
    """
    if not context:
        return {}

    system_prompt = (
        "You help assemble a visa checklist for Pakistani applicants. "
        "You will be given excerpts from an official visa guide. "
        "Using ONLY information present in the excerpts, return a JSON object "
        "mapping each of these category names to a short verified detail note "
        "(1-2 sentences, include concrete figures/fees/thresholds if present): "
        "Identity, Application, Purpose of Travel, Financial Evidence, "
        "Accommodation, Insurance, Academic. "
        "If the excerpts say nothing relevant to a category, omit that key. "
        "Never invent a figure or rule that is not in the excerpts. "
        "Return JSON only, no other text."
    )
    user_prompt = (
        f"Destination: {country}\nVisa type: {visa_type}\n\n"
        f"Official guide excerpts:\n{context}"
    )
    raw = ask_groq(system_prompt, user_prompt, max_tokens=700)
    parsed = _extract_json_block(raw)
    return parsed if isinstance(parsed, dict) else {}


# ---- Document field extraction ----------------------------------------
def extract_fields_with_llm(document_text: str, category: str):
    system_prompt = (
        "You extract structured fields from a visa applicant's supporting "
        "document. Only use information literally present in the text; use "
        "null for anything absent. Mask all but the last 4 characters of any "
        "ID or passport number. Return strict JSON with these keys: "
        "document_type, full_name, id_or_passport_number, issue_date, "
        "expiry_date, key_amount, issuer_or_institution, notes."
    )
    user_prompt = f"Document category: {category}\n\nDocument text:\n{document_text[:6000]}"
    raw = ask_groq(system_prompt, user_prompt, max_tokens=400)
    parsed = _extract_json_block(raw)
    if isinstance(parsed, dict):
        return parsed
    return {
        "document_type": category, "full_name": None, "id_or_passport_number": None,
        "issue_date": None, "expiry_date": None, "key_amount": None,
        "issuer_or_institution": None, "notes": "Automatic extraction unavailable; please fill in manually.",
    }


# ---- Next-step guidance -------------------------------------------------
def generate_copilot_summary(profile, score, blocking, review_items):
    system_prompt = (
        "You are a calm, encouraging visa-preparation assistant. Write 2-3 "
        "short sentences summarizing the applicant's current readiness and "
        "the single most important thing to do next. Do not invent visa "
        "rules. Do not guarantee approval outcomes."
    )
    user_prompt = json.dumps({
        "country": profile.get("country"),
        "visa_type": profile.get("visa_type"),
        "readiness_score": score,
        "blocking_issue_count": len(blocking),
        "review_item_count": len(review_items),
        "top_blocking_issue": blocking[0]["label"] if blocking else None,
    })
    text = ask_groq(system_prompt, user_prompt, max_tokens=200, temperature=0.4)
    return text


# ========================================================================
# SESSION STATE
# ========================================================================
def init_state():
    defaults = {
        "page": "Applicant Profile",
        "profile": {
            "country": None, "visa_type": None, "travel_date": None,
            "application_city": "", "first_time_applicant": None,
            "funding_source": "", "accommodation_arranged": None,
            "previous_schengen_travel": None,
        },
        "documents": {},   # doc_id -> {name, category, text, extracted, confirmed, manual}
        "checklist_notes": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def profile_completeness():
    p = st.session_state.profile
    fields = ["country", "visa_type", "travel_date", "application_city",
              "first_time_applicant", "funding_source", "accommodation_arranged"]
    filled = sum(1 for f in fields if p.get(f) not in (None, ""))
    return filled / len(fields)


def active_checklist():
    """Returns baseline checklist items relevant to the selected visa type."""
    visa_type = st.session_state.profile.get("visa_type")
    if not visa_type:
        return []
    return [item for item in BASELINE_CHECKLIST if item["visa"] in ("All", visa_type)]


def mandatory_items():
    return [i for i in active_checklist() if i["mandatory"]]


def docs_for_item(item_id):
    return [d for d in st.session_state.documents.values() if d["category"] == item_id]


def compute_issues():
    """Returns a list of issue dicts: {label, detail, severity}."""
    issues = []
    profile = st.session_state.profile

    # 1. Missing mandatory checklist items
    for item in mandatory_items():
        if not docs_for_item(item["id"]):
            issues.append({
                "label": f"Missing: {item['label']}",
                "detail": f"No document has been uploaded yet for '{item['label']}'.",
                "severity": "Action Required",
                "type": "missing",
                "item_id": item["id"],
            })

    # 2. Unconfirmed extractions
    for doc in st.session_state.documents.values():
        if not doc.get("confirmed"):
            issues.append({
                "label": f"Confirm extracted data: {doc['name']}",
                "detail": "AI-extracted details for this document have not been reviewed and confirmed yet.",
                "severity": "Review Recommended",
                "type": "unconfirmed",
                "doc_id": doc["id"],
            })

    # 3. Name consistency across confirmed documents
    names = []
    for doc in st.session_state.documents.values():
        if doc.get("confirmed"):
            name = (doc.get("extracted") or {}).get("full_name")
            if name:
                names.append((doc["name"], name.strip().lower()))
    if len(names) >= 2:
        base = names[0][1]
        for doc_name, name in names[1:]:
            if name != base and name not in base and base not in name:
                issues.append({
                    "label": "Name mismatch across documents",
                    "detail": f"'{names[0][0]}' shows a different name than '{doc_name}'. Please verify.",
                    "severity": "Action Required",
                    "type": "inconsistency",
                })
                break

    # 4. Passport expiry vs travel date (simple heuristic)
    travel_date = profile.get("travel_date")
    for doc in st.session_state.documents.values():
        if doc["category"] == "passport" and doc.get("confirmed"):
            expiry = (doc.get("extracted") or {}).get("expiry_date")
            if expiry and travel_date:
                try:
                    expiry_dt = datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date()
                    if (expiry_dt - travel_date).days < 90:
                        issues.append({
                            "label": "Passport may not be valid long enough",
                            "detail": "Your passport should generally remain valid for at least 3 months beyond your travel date. Please double-check the expiry date.",
                            "severity": "Action Required",
                            "type": "expiry",
                        })
                except Exception:
                    pass

    return issues


def compute_score():
    checklist = active_checklist()
    mandatory = mandatory_items()

    # Mandatory requirements completion
    if mandatory:
        completed_mandatory = sum(1 for i in mandatory if docs_for_item(i["id"]))
        mandatory_completion = completed_mandatory / len(mandatory)
    else:
        mandatory_completion = 0.0

    # Confirmed extraction
    docs = list(st.session_state.documents.values())
    confirmed_extraction = (sum(1 for d in docs if d.get("confirmed")) / len(docs)) if docs else 0.0

    # Consistency: 1 minus proportion of inconsistency-type issues among confirmed docs
    issues = compute_issues()
    inconsistency_issues = [i for i in issues if i["type"] in ("inconsistency", "expiry")]
    consistency_score = 1.0 if not inconsistency_issues else max(0.0, 1 - 0.5 * len(inconsistency_issues))

    # Applicant profile completeness
    profile_score = profile_completeness()

    weighted = (
        0.40 * mandatory_completion +
        0.25 * confirmed_extraction +
        0.20 * consistency_score +
        0.15 * profile_score
    )
    overall = round(weighted * 100)

    components = {
        "Mandatory Requirements": round(mandatory_completion * 100),
        "Confirmed Extraction": round(confirmed_extraction * 100),
        "Consistency": round(consistency_score * 100),
        "Applicant Profile": round(profile_score * 100),
    }
    return overall, components, issues


# ========================================================================
# PAGE RENDERERS
# ========================================================================
def page_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<div class='vp-muted'>{subtitle}</div>", unsafe_allow_html=True)
    st.write("")


def render_profile():
    page_header("Applicant Profile", "Tell Viza Pilot where you're going so it can prepare the right checklist.")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox(
                "Destination country",
                list(COUNTRIES.keys()),
                index=list(COUNTRIES.keys()).index(st.session_state.profile["country"])
                if st.session_state.profile["country"] in COUNTRIES else 0,
            )
            visa_type = st.selectbox(
                "Visa type",
                VISA_TYPES,
                index=VISA_TYPES.index(st.session_state.profile["visa_type"])
                if st.session_state.profile["visa_type"] in VISA_TYPES else 0,
            )
            travel_date = st.date_input(
                "Expected travel / intake date",
                value=st.session_state.profile["travel_date"] or date.today(),
            )
            application_city = st.selectbox(
                "Where will you apply?",
                ["Islamabad", "Karachi", "Lahore", "Multan", "Other"],
                index=0,
            )
        with col2:
            first_time = st.radio("Is this your first visa application to this country?", ["Yes", "No"], horizontal=True)
            funding_source = st.selectbox(
                "How will you fund your stay?",
                ["Personal savings", "Sponsor / family", "Scholarship", "Blocked account", "Employer-funded"],
            )
            accommodation = st.radio("Do you have accommodation arranged?", ["Yes", "In progress", "Not yet"], horizontal=True)
            previous_schengen = None
            if COUNTRIES[country]["schengen"]:
                previous_schengen = st.radio("Any Schengen travel in the past 180 days?", ["Yes", "No"], horizontal=True)

        if not COUNTRIES[country]["guide_file"]:
            st.info(
                f"Viza Pilot doesn't yet have a verified official source loaded for **{country}**. "
                "You'll still get a general baseline checklist, but please confirm exact requirements "
                "with the relevant embassy, consulate, or visa application centre."
            )

        if st.button("Save profile and continue", type="primary"):
            st.session_state.profile.update({
                "country": country,
                "visa_type": visa_type,
                "travel_date": travel_date,
                "application_city": application_city,
                "first_time_applicant": first_time,
                "funding_source": funding_source,
                "accommodation_arranged": accommodation,
                "previous_schengen_travel": previous_schengen,
            })
            st.session_state.page = "Visa Checklist"
            st.rerun()


def render_checklist():
    page_header("Visa Checklist", "Your personalized list of required documents.")

    profile = st.session_state.profile
    if not profile.get("country") or not profile.get("visa_type"):
        st.warning("Please complete your Applicant Profile first.")
        return

    country = profile["country"]
    has_guide = bool(COUNTRIES[country]["guide_file"])
    items = active_checklist()

    total_mandatory = len(mandatory_items())
    completed_mandatory = sum(1 for i in mandatory_items() if docs_for_item(i["id"]))
    st.progress(completed_mandatory / total_mandatory if total_mandatory else 0)
    st.caption(f"{completed_mandatory} of {total_mandatory} mandatory requirements have a document uploaded.")

    if has_guide:
        model, kb = load_knowledge_base()
        context = gather_country_context(country, model, kb)
        notes = generate_verified_notes(country, profile["visa_type"], context)
    else:
        notes = {}
        st.markdown(
            "<span class='vp-badge-warn'>General guidance — not yet verified against an official source</span>",
            unsafe_allow_html=True,
        )
        st.write("")

    categories = {}
    for item in items:
        categories.setdefault(item["category"], []).append(item)

    for category, cat_items in categories.items():
        with st.container():
            st.markdown(f"#### {category}")
            note = notes.get(category)
            if note:
                st.markdown(
                    f"<div class='vp-card'><span class='vp-badge-ok'>Verified detail</span>"
                    f"<p class='vp-muted' style='margin-top:0.5rem'>{note}</p></div>",
                    unsafe_allow_html=True,
                )
            for item in cat_items:
                done = bool(docs_for_item(item["id"]))
                tag = "✅" if done else ("🔴" if item["mandatory"] else "⚪")
                req_tag = "Mandatory" if item["mandatory"] else "Recommended"
                st.markdown(f"{tag} **{item['label']}**  \n<span class='vp-muted'>{req_tag}</span>", unsafe_allow_html=True)
            st.write("")

    st.button("Continue to Upload Documents", type="primary",
              on_click=lambda: st.session_state.update(page="Upload Documents"))


def render_upload():
    page_header("Upload Documents", "Upload a PDF, text file, scan, or phone photo. Viza Pilot reads both digital text and image-based documents automatically.")

    if not _tesseract_available():
        st.warning(
            "Scanned PDFs and photos require Tesseract OCR on the server. "
            "Digital PDFs and text files will still work."
        )

    items = active_checklist()
    if not items:
        st.warning("Please complete your Applicant Profile first.")
        return

    for item in items:
        with st.expander(f"{item['label']}  {'(Mandatory)' if item['mandatory'] else '(Recommended)'}", expanded=False):
            uploaded = st.file_uploader(
                "Upload file", type=["pdf", "txt", "png", "jpg", "jpeg"],
                key=f"upload_{item['id']}", label_visibility="collapsed",
            )
            if uploaded is not None:
                doc_id = f"{item['id']}_{uploaded.name}"
                if doc_id not in st.session_state.documents:
                    with st.spinner("Reading document..."):
                        text, read_method, read_error = _extract_uploaded_text(uploaded)

                    extracted = {}
                    if text.strip():
                        with st.spinner("Extracting key details..."):
                            extracted = extract_fields_with_llm(text, item["label"])
                        if read_method == "OCR":
                            extracted["notes"] = (extracted.get("notes") or "") + " Read from the uploaded image."
                        elif read_method == "OCR + PDF text":
                            extracted["notes"] = (extracted.get("notes") or "") + " Scanned PDF page(s) were read automatically."
                    else:
                        extracted = {
                            "document_type": item["label"], "full_name": None,
                            "id_or_passport_number": None, "issue_date": None,
                            "expiry_date": None, "key_amount": None,
                            "issuer_or_institution": None,
                            "notes": read_error or "No readable text found — please enter details manually on the Review tab.",
                        }

                    st.session_state.documents[doc_id] = {
                        "id": doc_id, "name": uploaded.name, "category": item["id"],
                        "label": item["label"], "text": text, "extracted": extracted,
                        "read_method": read_method, "confirmed": False,
                    }
                st.success(f"Uploaded: {uploaded.name} — go to **Review Extracted Data** to confirm details.")

            existing = docs_for_item(item["id"])
            for doc in existing:
                badge = "vp-badge-ok" if doc["confirmed"] else "vp-badge-warn"
                label = "Confirmed" if doc["confirmed"] else "Needs your review"
                st.markdown(f"<span class='{badge}'>{label}</span> &nbsp; {doc['name']}", unsafe_allow_html=True)

    st.button("Continue to Review Extracted Data", type="primary",
              on_click=lambda: st.session_state.update(page="Review Extracted Data"))


def render_review():
    page_header("Review Extracted Data", "Viza Pilot reads each document automatically — please confirm the details are correct.")

    docs = list(st.session_state.documents.values())
    if not docs:
        st.info("No documents uploaded yet. Go to the Upload Documents tab first.")
        return

    for doc in docs:
        with st.container():
            st.markdown(f"#### {doc['label']}")
            st.caption(doc["name"])
            badge = "vp-badge-ok" if doc["confirmed"] else "vp-badge-warn"
            label = "Confirmed" if doc["confirmed"] else "AI Extracted — Please Confirm"
            st.markdown(f"<span class='{badge}'>{label}</span>", unsafe_allow_html=True)
            st.write("")

            extracted = doc.get("extracted") or {}
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full name", value=extracted.get("full_name") or "", key=f"name_{doc['id']}")
                id_number = st.text_input("ID / passport number", value=extracted.get("id_or_passport_number") or "", key=f"id_{doc['id']}")
                issue_date = st.text_input("Issue date", value=extracted.get("issue_date") or "", key=f"issue_{doc['id']}")
            with col2:
                expiry_date = st.text_input("Expiry date (YYYY-MM-DD if known)", value=extracted.get("expiry_date") or "", key=f"expiry_{doc['id']}")
                key_amount = st.text_input("Key amount (if financial document)", value=extracted.get("key_amount") or "", key=f"amount_{doc['id']}")
                issuer = st.text_input("Issuer / institution", value=extracted.get("issuer_or_institution") or "", key=f"issuer_{doc['id']}")

            if extracted.get("notes"):
                st.caption(f"Note: {extracted['notes']}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm this is correct", key=f"confirm_{doc['id']}", type="primary"):
                    doc["extracted"] = {
                        "document_type": doc["label"], "full_name": full_name,
                        "id_or_passport_number": id_number, "issue_date": issue_date,
                        "expiry_date": expiry_date, "key_amount": key_amount,
                        "issuer_or_institution": issuer, "notes": "",
                    }
                    doc["confirmed"] = True
                    st.rerun()
            with c2:
                if st.button("Remove this document", key=f"remove_{doc['id']}"):
                    del st.session_state.documents[doc["id"]]
                    st.rerun()
            st.divider()

    st.button("Continue to Issue Center", type="primary",
              on_click=lambda: st.session_state.update(page="Issue Center"))


def render_issues():
    page_header("Issue Center", "Everything that needs your attention before you submit.")

    issues = compute_issues()
    blocking = [i for i in issues if i["severity"] == "Action Required"]
    review = [i for i in issues if i["severity"] == "Review Recommended"]

    st.markdown(f"**{len(blocking)} issue(s) need your attention, {len(review)} need a quick review.**")
    st.write("")

    if blocking:
        st.markdown("#### Action Required")
        for issue in blocking:
            st.markdown(
                f"<div class='vp-card'><span class='vp-badge-risk'>Action Required</span>"
                f"<p style='margin-top:0.5rem'><b>{issue['label']}</b></p>"
                f"<p class='vp-muted'>{issue['detail']}</p></div>",
                unsafe_allow_html=True,
            )
    if review:
        st.markdown("#### Review Recommended")
        for issue in review:
            st.markdown(
                f"<div class='vp-card'><span class='vp-badge-warn'>Review Recommended</span>"
                f"<p style='margin-top:0.5rem'><b>{issue['label']}</b></p>"
                f"<p class='vp-muted'>{issue['detail']}</p></div>",
                unsafe_allow_html=True,
            )
    if not issues:
        st.success("No open issues right now. Nice work!")

    st.button("See my Readiness Score", type="primary",
              on_click=lambda: st.session_state.update(page="Readiness Score"))


def render_score():
    page_header("Readiness Score", "How ready is your application, right now?")

    overall, components, issues = compute_score()
    blocking = [i for i in issues if i["severity"] == "Action Required"]
    review = [i for i in issues if i["severity"] == "Review Recommended"]

    if overall >= 80:
        status, badge = "Almost Ready", "vp-badge-ok"
    elif overall >= 50:
        status, badge = "In Progress", "vp-badge-warn"
    else:
        status, badge = "Needs Work", "vp-badge-risk"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<p class='vp-score'>{overall}</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='{badge}'>{status}</span>", unsafe_allow_html=True)
    with col2:
        st.write("")
        for name, value in components.items():
            st.caption(f"{name} — {value}%")
            st.progress(value / 100)

    st.divider()

    st.markdown("#### Required Checklist Items")
    for item in mandatory_items():
        done = bool(docs_for_item(item["id"]))
        st.markdown(f"{'✅' if done else '⬜️'} {item['label']}")

    st.markdown("#### Blocking Issues")
    if blocking:
        for issue in blocking:
            st.markdown(f"- {issue['label']}")
    else:
        st.caption("None right now.")

    st.markdown("#### Items Requiring Review")
    if review:
        for issue in review:
            st.markdown(f"- {issue['label']}")
    else:
        st.caption("None right now.")

    st.write("")
    st.button("What should I do next?", type="primary",
              on_click=lambda: st.session_state.update(page="What To Do Next"))


def render_next_steps():
    page_header("What To Do Next", "Your prioritized action list.")

    overall, components, issues = compute_score()
    blocking = [i for i in issues if i["severity"] == "Action Required"]
    review = [i for i in issues if i["severity"] == "Review Recommended"]

    summary = generate_copilot_summary(st.session_state.profile, overall, blocking, review)
    if summary:
        st.markdown(f"<div class='vp-card'>{summary}</div>", unsafe_allow_html=True)

    actions = []
    for issue in blocking:
        actions.append(("High", issue["label"], "~5 min"))
    for issue in review:
        actions.append(("Medium", issue["label"], "~3 min"))

    if not actions:
        st.success("You're in great shape. Review your documents once more, then proceed with your official application.")
    else:
        st.markdown("#### Priority actions")
        for priority, label, est in actions:
            tag = "vp-badge-risk" if priority == "High" else "vp-badge-warn"
            st.markdown(
                f"<div class='vp-card'><span class='{tag}'>{priority} priority</span>"
                f"&nbsp;&nbsp;<b>{label}</b>&nbsp;&nbsp;<span class='vp-muted'>{est}</span></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption("Once every mandatory item is uploaded, confirmed, and consistent, you'll be ready to book your appointment through the official portal for your destination.")


# ========================================================================
# FEEDBACK WIDGET (bottom-right corner): visits, AI usage, rating
# ========================================================================
def render_feedback_widget():
    stats = load_stats()
    avg_rating = (stats["rating_sum"] / stats["rating_count"]) if stats["rating_count"] else None
    rating_text = f"⭐ {avg_rating:.1f} ({stats['rating_count']})" if avg_rating else "No ratings yet"

    with st.container(key="vp_feedback_widget"):
        st.markdown(
            f"<div class='vp-stats-line'>{stats['visits']:,} visits &nbsp;·&nbsp; "
            f"{stats['total_tokens']:,} AI tokens used &nbsp;·&nbsp; {rating_text}</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.get("rating_submitted"):
            st.caption("Thanks for your feedback! 🙏")
        else:
            st.markdown("<div class='vp-rate-label'>Rate Viza Pilot</div>", unsafe_allow_html=True)
            rating = st.feedback("stars", key="vp_rating_widget")
            if rating is not None:
                add_rating(rating + 1)  # st.feedback("stars") returns 0-4
                st.session_state["rating_submitted"] = True
                st.rerun()


# ========================================================================
# MAIN
# ========================================================================
def main():
    inject_style()
    init_state()

    # Count each new browser session once, not on every rerun.
    if not st.session_state.get("visit_counted"):
        st.session_state["visit_counted"] = True
        increment_visits()

    with st.sidebar:
        st.markdown("### 🧭 Viza Pilot")
        st.caption("Apply smarter. Travel prepared.")
        st.write("")

        pages = [
            "Applicant Profile",
            "Visa Checklist",
            "Upload Documents",
            "Review Extracted Data",
            "Issue Center",
            "Readiness Score",
            "What To Do Next",
        ]
        current = st.session_state.page if st.session_state.page in pages else pages[0]
        choice = st.radio("Navigate", pages, index=pages.index(current), label_visibility="collapsed")
        st.session_state.page = choice

        st.write("")
        st.divider()
        profile = st.session_state.profile
        if profile.get("country"):
            st.caption(f"Destination: **{profile['country']}**")
            st.caption(f"Visa type: **{profile.get('visa_type', '—')}**")
        if st.button("Start over"):
            for key in ["profile", "documents", "checklist_notes"]:
                del st.session_state[key]
            st.session_state.page = "Applicant Profile"
            init_state()
            st.rerun()

    routes = {
        "Applicant Profile": render_profile,
        "Visa Checklist": render_checklist,
        "Upload Documents": render_upload,
        "Review Extracted Data": render_review,
        "Issue Center": render_issues,
        "Readiness Score": render_score,
        "What To Do Next": render_next_steps,
    }
    routes[st.session_state.page]()

    st.markdown(f"<div class='vp-footer-disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    render_feedback_widget()


if __name__ == "__main__":
    main()
