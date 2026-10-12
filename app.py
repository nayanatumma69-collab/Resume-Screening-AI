import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Screening & Candidate Ranking")
st.write(
    "Upload resumes and compare their text with a job description using "
    "Sentence-BERT semantic similarity."
)
st.info(
    "This is a decision-support demo, not an automated hiring decision. "
    "Review each resume yourself. Match scores are approximate text-similarity "
    "signals, not probabilities of job performance."
)

SKILL_DATABASE = [
    "python", "machine learning", "deep learning", "nlp",
    "natural language processing", "pytorch", "tensorflow",
    "scikit-learn", "huggingface", "transformers", "pandas", "numpy",
    "docker", "kubernetes", "sql", "flask", "fastapi", "streamlit",
    "aws", "azure", "gcp", "git", "data analysis", "tableau",
    "power bi", "llm", "opencv",
]

EDUCATION_PATTERNS = [
    r"\b(?:B\.?Tech|B\.?E\.?|B\.?S\.?|B\.?Sc|Bachelor(?:'s)?(?:\s+of\s+[\w\s]+)?)\b",
    r"\b(?:M\.?Tech|M\.?S\.?|M\.?Sc|Master(?:'s)?(?:\s+of\s+[\w\s]+)?)\b",
    r"\b(?:Ph\.?D|Doctorate)\b",
    r"\b(?:BBA|MBA|BCA|MCA)\b",
]
EXPERIENCE_PATTERNS = [
    r"(\d+(?:\.\d+)?\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?)",
    r"(\b\d+\+?\s*(?:years?|yrs?)\b)",
]

DEFAULT_JD = """Job Title: AI & Machine Learning Engineer

Role Overview:
We are seeking an AI Engineer proficient in Python, Machine Learning,
Deep Learning, and Natural Language Processing (NLP).

Key Requirements & Skills:
- PyTorch, TensorFlow, HuggingFace Transformers, and Scikit-learn
- Pandas, NumPy, and SQL
- Flask, FastAPI, Streamlit, and deployment of ML models
- Bachelor's or Master's degree in Computer Science, Data Science, or a related field
- Experience developing, fine-tuning, and deploying machine learning models
"""

@st.cache_resource(show_spinner="Loading Sentence-BERT model (first load may take a few minutes)...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(uploaded_file):
    """Extract text from a PDF or TXT upload without saving it to disk."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            uploaded_file.seek(0)
            reader = PyPDF2.PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages).strip()
        if name.endswith(".txt"):
            uploaded_file.seek(0)
            return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        return f"ERROR: Could not read {uploaded_file.name}: {exc}"
    return ""

def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in SKILL_DATABASE:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill.title())
    return sorted(found)

def extract_education(text):
    found = set()
    for pattern in EDUCATION_PATTERNS:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            found.add(match.strip())
    return sorted(found) if found else ["Not explicitly identified"]

def extract_experience(text):
    found = []
    for pattern in EXPERIENCE_PATTERNS:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    unique = list(dict.fromkeys(item.strip() for item in found))
    return unique[:3] if unique else ["Not explicitly specified"]

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Extract text from PDF/TXT resumes\n"
        "2. Identify listed skills, education, and stated experience\n"
        "3. Encode the job description and resumes with Sentence-BERT\n"
        "4. Rank by cosine similarity\n"
        "5. Download a CSV report"
    )
    st.caption("Uploaded resume text is processed in memory by this app and is not intentionally saved to a file.")

col_jd, col_upload = st.columns([1, 1])
with col_jd:
    st.subheader("1. Job description")
    job_description = st.text_area(
        "Paste the target job description",
        value=DEFAULT_JD,
        height=300,
    )
with col_upload:
    st.subheader("2. Upload candidate resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="For best results, use text-based PDFs. Scanned image-only PDFs may need OCR.",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected.")

if st.button("🚀 Screen and rank candidates", type="primary", use_container_width=True):
    if not job_description.strip():
        st.error("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        parsed_candidates = []
        for file in uploaded_files:
            text = extract_text(file)
            if text.startswith("ERROR:"):
                st.error(text)
                continue
            if not text:
                st.warning(f"No readable text was extracted from {file.name}. It may be scanned or empty.")
                continue
            parsed_candidates.append({
                "name": Path(file.name).stem.replace("_", " ").replace("-", " ").title(),
                "filename": file.name,
                "text": text,
                "skills": extract_skills(text),
                "education": extract_education(text),
                "experience": extract_experience(text),
            })

        if not parsed_candidates:
            st.error("No readable resumes were found. Try text-based PDF or TXT files.")
        else:
            try:
                model = get_embedder()
                with st.spinner("Comparing job description and resumes..."):
                    texts = [job_description] + [candidate["text"] for candidate in parsed_candidates]
                    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                    jd_embedding = embeddings[0]
                    for idx, candidate in enumerate(parsed_candidates, start=1):
                        similarity = float(np.dot(jd_embedding, embeddings[idx]))
                        candidate["score"] = round(float(np.clip(similarity, 0.0, 1.0)) * 100, 2)

                parsed_candidates.sort(key=lambda item: item["score"], reverse=True)
                report_rows = []
                st.subheader("📊 Candidate ranking")
                for rank, candidate in enumerate(parsed_candidates, start=1):
                    with st.expander(
                        f"#{rank} — {candidate['name']} · {candidate['score']:.2f}% text similarity",
                        expanded=(rank == 1),
                    ):
                        st.progress(candidate["score"] / 100.0)
                        c1, c2, c3 = st.columns(3)
                        c1.markdown("**Extracted skills**")
                        c1.write(", ".join(candidate["skills"]) or "None detected")
                        c2.markdown("**Education**")
                        c2.write(", ".join(candidate["education"]))
                        c3.markdown("**Experience stated in resume**")
                        c3.write(", ".join(candidate["experience"]))
                        with st.expander("Preview extracted resume text"):
                            st.text(candidate["text"][:5000])

                    report_rows.append({
                        "Rank": rank,
                        "Candidate Name": candidate["name"],
                        "File Name": candidate["filename"],
                        "Text Similarity (%)": candidate["score"],
                        "Extracted Skills": ", ".join(candidate["skills"]),
                        "Education": ", ".join(candidate["education"]),
                        "Experience": ", ".join(candidate["experience"]),
                    })

                report_df = pd.DataFrame(report_rows)
                st.subheader("📑 Downloadable report")
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download candidate ranking report (CSV)",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name="candidate_ranking_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption(
                    "Scores reflect semantic text similarity only. They can be affected by resume wording, "
                    "formatting, and the selected job description; they should not be used as the sole hiring criterion."
                )
            except Exception as exc:
                st.error(f"Could not complete resume screening: {exc}")
                st.info("If this is the first launch, wait for the model to finish downloading and try again.")
