# AI Resume Screening & Candidate Ranking

A Streamlit app adapted from the uploaded `Resume_Screening_AI.ipynb` notebook. It extracts text from PDF/TXT resumes, detects a predefined list of skills, education, and stated experience, and compares resume text with a job description using Sentence-BERT (`all-MiniLM-L6-v2`) and cosine similarity.

## Features
- Upload multiple PDF or TXT resumes
- Edit the target job description
- Extract skills, education, and stated experience using pattern matching
- Semantic text similarity and candidate ranking using Sentence-BERT
- Download the results as CSV
- Streamlit UI for free deployment

## Important limitations and responsible use
- The score is a text-similarity measure, not a probability of job performance or suitability.
- The skill and education extractors use predefined patterns and can miss information or produce imperfect matches.
- Do not use this tool as the sole basis for hiring decisions. A human should review the original resume and the context of each score.
- Avoid uploading resumes without an appropriate lawful basis and consent. The app does not intentionally save uploaded resume files, but files are processed by the hosted application while it runs.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free
1. Create a GitHub repository named `ai-resume-screening`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore` to the repository root.
3. Open https://share.streamlit.io and sign in with GitHub.
4. Choose **Create app**, select the repository and `main` branch, and set the app file path to `app.py`.
5. Click **Deploy**.

The first launch downloads the Sentence-BERT model, so startup can take several minutes. If the free host runs out of memory or times out, we may need to reduce dependencies or use a lighter model.

The original notebook is preserved in `notebooks/` for reference.

## Project structure
- `app.py` — Streamlit dashboard and ranking logic
- `requirements.txt` — Python dependencies
- `notebooks/Resume_Screening_AI.ipynb` — original Colab notebook
