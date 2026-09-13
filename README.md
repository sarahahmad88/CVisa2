# 🧭 Viza Pilot

**Viza Pilot** is an AI visa preparation co-pilot for Pakistani passport holders applying for Student or Tourist visas. It walks you from an applicant profile through a personalized checklist, document upload, AI-assisted data review, an issue center, a readiness score, and a prioritized "what to do next" list.

Under the hood it's a **hybrid RAG (retrieval-augmented generation) application**:

| Layer | Tool |
|---|---|
| Interface | Streamlit |
| Document reading (official guides + your uploads) | PyMuPDF |
| Semantic search embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| Language generation (checklist notes, extraction, guidance) | Groq (Llama 3.3) |

None of this is exposed to the end user — the interface only ever shows plain checklist items, extracted document fields, issues, and a readiness score.

## Why "hybrid"?

Every specific figure or rule shown in the checklist (fees, financial thresholds, insurance minimums, procedures) is **retrieved from the bundled official-guide text** for that country before the model is allowed to phrase it — the model is instructed never to invent a requirement that isn't in the retrieved text. Countries without a bundled guide clearly show a "not yet verified" notice instead of a fabricated checklist.

## Features

- **Applicant Profile** — destination, visa type, travel date, funding, accommodation status
- **Visa Checklist** — personalized, category-grouped checklist with verified detail notes where a guide is available
- **Upload Documents** — per-requirement upload (PDF/text auto-read; images can be filled in manually)
- **Review Extracted Data** — "AI Extracted — Please Confirm" pattern; nothing is used until you confirm or correct it
- **Issue Center** — missing mandatory items, unconfirmed extractions, and cross-document inconsistencies, grouped by severity
- **Readiness Score** — a 0–100 score built from four components: Mandatory Requirements, Confirmed Extraction, Consistency, and Applicant Profile completeness, plus a summary of required items, blocking issues, and items needing review
- **What To Do Next** — a prioritized action list to raise your readiness
- **Feedback widget** (bottom-right corner, on every page) — a small floating panel showing total visits, AI tokens used, and the average user rating, plus a one-click 1–5 star rating

Countries included: Germany, France, Italy (official guides bundled and verified), plus Spain, Greece, Switzerland, Norway, United Kingdom, and Turkey (general baseline checklist, clearly marked as not yet source-verified).

> Viza Pilot does not make visa decisions and cannot guarantee approval. Always confirm current requirements with the relevant embassy, consulate, or visa application centre.

## Project structure

```
vizapilot/
├── app.py                # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md
├── .gitignore
└── knowledge/             # Bundled official-guide source text (used for retrieval)
    ├── Germany_Visa_Guide_Pakistan.txt
    ├── France_Visa_Guide_Pakistan.txt
    └── Italy_Visa_Guide_Pakistan.txt
```

You can also drop the **original PDF guides** into `knowledge/` (same file name, `.pdf` extension) — the app reads either `.pdf` or `.txt` automatically via PyMuPDF.

---

## Deploy it — no terminal, no VS Code, no Colab required

Everything below is done entirely through your web browser: the GitHub website and the Streamlit Community Cloud website.

### Step 1 — Create a GitHub account (skip if you have one)

1. Go to [github.com](https://github.com) and sign up.

### Step 2 — Create a new repository

1. Click the **+** icon (top right) → **New repository**.
2. Name it, e.g. `vizapilot`.
3. Set it to **Public** (Streamlit Community Cloud's free tier deploys public repos most simply).
4. Leave "Add a README" **unchecked** (you'll upload the provided one).
5. Click **Create repository**.

### Step 3 — Upload the project files

1. On your new (empty) repository page, click **"uploading an existing file"** (or **Add file → Upload files**).
2. Drag and drop these four files from your computer into the upload box:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Scroll down and click **Commit changes**.

### Step 4 — Upload the knowledge base folder

1. On the repository page, click **Add file → Upload files** again.
2. This time, drag the entire **`knowledge`** folder (containing the three `.txt` guide files) from your computer directly into the upload box. GitHub's drag-and-drop preserves the folder structure, so it will land at `knowledge/…` in your repo automatically.
3. Click **Commit changes**.
4. Confirm you now see an `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and a `knowledge/` folder with three files in your repository's file list.

### Step 5 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com) and sign up (free).
2. Open **API Keys** in the left menu → **Create API Key**.
3. Copy the key (it starts with `gsk_...`). You won't be able to see it again, so keep the tab open or paste it somewhere safe for the next step.

### Step 6 — Create your Streamlit Community Cloud account

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your GitHub account and authorize Streamlit to access your repositories.

### Step 7 — Deploy the app

1. Click **Create app** (or **New app**).
2. Choose **"Deploy a public app from GitHub"**.
3. Select:
   - **Repository:** `your-username/vizapilot`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** before deploying:
   - Under **Secrets**, paste:
     ```
     GROQ_API_KEY = "gsk_your_actual_key_here"
     ```
   - You can leave the Python version at its default.
5. Click **Deploy**.

Streamlit Cloud will now build the app (installing everything in `requirements.txt`, which can take a few minutes the first time because of the embedding model dependencies). When it finishes, your app opens automatically at a URL like:

```
https://your-username-vizapilot.streamlit.app
```

### Step 8 — Test it

1. Open the app URL.
2. Go through **Applicant Profile → Visa Checklist** for Germany, France, or Italy and confirm you see verified detail notes (these come from the bundled guides).
3. Try **Upload Documents** with a sample PDF, then **Review Extracted Data** to confirm the extracted fields.
4. Check the **Readiness Score** and **What To Do Next** tabs update as you complete items.

### Updating the app later

Any time you want to change something:
1. Open the file on GitHub (e.g. `app.py`), click the pencil (✏️) icon to edit directly in the browser, or use **Add file → Upload files** to replace it.
2. Commit the change.
3. Streamlit Cloud automatically redeploys within a minute or two — no manual redeploy step needed.

### Adding more countries later

To add a verified guide for another country (e.g. Spain):
1. Add the guide as `knowledge/Spain_Visa_Guide_Pakistan.txt` (or `.pdf`) via **Add file → Upload files** on GitHub.
2. In `app.py`, find the `COUNTRIES` dictionary and change Spain's entry to:
   ```python
   "Spain": {"guide_file": "Spain_Visa_Guide_Pakistan", "schengen": True},
   ```
3. Commit the change — Streamlit Cloud will redeploy and Spain will now use the verified checklist path.

## Notes and limitations (by design, for this MVP)

- Data lives only in your browser session (`st.session_state`) — nothing is persisted to a database. Refreshing the page clears your progress. This keeps the MVP simple and avoids storing sensitive documents long-term.
- Image uploads (JPG/PNG) are not OCR'd in this MVP — you can still complete the Review tab manually for those.
- The Readiness Score and Risk indicators reflect **documentation completeness and consistency only** — they are not a prediction of an immigration authority's decision.
- **Usage stats (visits / AI tokens / ratings) are stored in a small local file** (`data/usage_stats.json`) rather than a database, so they're shared across everyone using the app while it stays running, but reset if the app is redeployed or wakes up from sleep on Streamlit Community Cloud. This is fine for an MVP; swap `load_stats()` / `save_stats()` in `app.py` for a small external store (e.g. a Google Sheet or hosted database) if you need the numbers to survive redeploys.
