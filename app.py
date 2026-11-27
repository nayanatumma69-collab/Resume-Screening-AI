import hashlib
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Display name -> spellings to look for in the text
SKILLS = {
    "Python": ["python"],
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Hugging Face": ["huggingface", "hugging face"],
    "Transformers": ["transformers"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes"],
    "SQL": ["sql"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Streamlit": ["streamlit"],
    "AWS": ["aws"],
    "Azure": ["azure"],
    "GCP": ["gcp"],
    "Git": ["git"],
    "Data Analysis": ["data analysis"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi"],
    "LLM": ["llm", "llms"],
    "OpenCV": ["opencv"],
}

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

STYLE = """
<style>
.hero { background: linear-gradient(120deg, #0f172a, #3730a3 55%, #0e7490);
        border-radius: 20px; padding: 30px 34px; margin-bottom: 20px; }
.hero h1 { color: #fff; margin: 0; padding: 0; font-size: 2rem; }
.hero p { color: #cbd5e1; margin: 8px 0 0; }
.grid { display: grid; gap: 14px; margin-bottom: 18px;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
.card { background: #fff; color: #0f172a; border-radius: 16px; padding: 18px;
        border: 1px solid #e2e8f0; box-shadow: 0 6px 18px rgba(15,23,42,.08); }
.stat { border-top: 5px solid var(--c); }
.stat small { color: #64748b; display: block; }
.stat b { font-size: 1.5rem; overflow-wrap: anywhere; }
.person { text-align: center; }
.ring { width: 140px; height: 140px; border-radius: 50%; margin: 6px auto 12px;
        display: grid; place-items: center;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), #e2e8f0 0); }
.ring span { width: 108px; height: 108px; border-radius: 50%; background: #fff;
             display: grid; place-items: center; font-size: 1.45rem; font-weight: 700; }
.person b { display: block; overflow-wrap: anywhere; }
.person small { color: #64748b; }
.chip { display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 999px;
        font-size: .85rem; font-weight: 600; }
.match { background: #d1fae5; color: #065f46; }
.missing { background: #ffe4e6; color: #9f1239; }
.extra { background: #e0e7ff; color: #3730a3; }
section[data-testid="stSidebar"] { border-right: 4px solid #4f46e5; }
</style>
"""

RING_COLORS = ["#4f46e5", "#0891b2", "#db2777", "#d97706", "#059669", "#7c3aed"]


@st.cache_resource(show_spinner="Loading Sentence-BERT model (first load may take a few minutes)...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(uploaded_file):
    """Extract text from a PDF or TXT upload without saving it to disk."""
    if uploaded_file.name.lower().endswith(".pdf"):
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()


def read_upload(uploaded_file):
    """Return the file's text, or an empty string after showing why it failed."""
    try:
        text = extract_text(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read {uploaded_file.name}: {exc}")
        return ""
    if not text:
        st.warning(f"No readable text in {uploaded_file.name}. It may be scanned or empty.")
    return text


def extract_skills(text):
    text = text.lower()
    return [
        skill for skill, spellings in SKILLS.items()
        if any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in spellings)
    ]


def extract_education(text):
    found = set()
    for pattern in EDUCATION_PATTERNS:
        found.update(match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE))
    return sorted(found) or ["Not explicitly identified"]


def extract_experience(text):
    found = []
    for pattern in EXPERIENCE_PATTERNS:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    unique = list(dict.fromkeys(item.strip() for item in found))
    return unique[:3] or ["Not explicitly specified"]


def rank_candidates(job_description, candidates):
    """Score each candidate against the job description and sort, best first."""
    texts = [job_description] + [c["text"] for c in candidates]
    embeddings = load_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    for candidate, embedding in zip(candidates, embeddings[1:]):
        similarity = float(np.dot(embeddings[0], embedding))
        candidate["score"] = round(float(np.clip(similarity, 0.0, 1.0)) * 100, 2)
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def ring_card(percent, color, title, subtitle):
    return (
        f'<div class="card person"><div class="ring" style="--p:{percent:.2f};--c:{color}">'
        f"<span>{percent:.1f}%</span></div><b>{html.escape(title)}</b>"
        f"<small>{html.escape(subtitle)}</small></div>"
    )


def stat_card(label, value, color):
    return (
        f'<div class="card stat" style="--c:{color}"><small>{label}</small>'
        f"<b>{html.escape(str(value))}</b></div>"
    )


def chips(skills, kind):
    if not skills:
        return "<em>None</em>"
    return "".join(f'<span class="chip {kind}">{html.escape(s)}</span>' for s in skills)


def show_html(content):
    st.markdown(content, unsafe_allow_html=True)


def grid(cards):
    show_html(f'<div class="grid">{"".join(cards)}</div>')


def show_results(candidates, jd_skills):
    top = candidates[0]
    average = sum(c["score"] for c in candidates) / len(candidates)
    grid([
        stat_card("👥 Candidates screened", len(candidates), "#4f46e5"),
        stat_card("🏆 Top ranked", top["name"], "#d97706"),
        stat_card("🎯 Highest similarity", f"{top['score']:.1f}%", "#db2777"),
        stat_card("📈 Average similarity", f"{average:.1f}%", "#0891b2"),
    ])

    overview, details, report = st.tabs(["🍩 Match overview", "🔍 Candidate details", "📑 Report"])

    with overview:
        st.subheader("Text similarity to the job description")
        grid([
            ring_card(c["score"], RING_COLORS[i % len(RING_COLORS)], c["name"], f"Rank #{i + 1}")
            for i, c in enumerate(candidates)
        ])
        if jd_skills:
            st.subheader("Job skills found in each resume")
            cards = []
            for i, c in enumerate(candidates):
                found = len(set(c["skills"]) & set(jd_skills))
                cards.append(ring_card(
                    100 * found / len(jd_skills), "#059669", c["name"],
                    f"{found} of {len(jd_skills)} skills",
                ))
            grid(cards)
            st.caption("Keyword check against the app's skill list. It does not affect the ranking.")

    with details:
        index = st.selectbox(
            "Choose a candidate",
            range(len(candidates)),
            format_func=lambda i: f"#{i + 1} · {candidates[i]['name']} ({candidates[i]['filename']})",
        )
        c = candidates[index]
        found = set(c["skills"])
        show_html(
            '<div class="card">'
            f"<p><b>✅ Matching skills</b><br>{chips([s for s in jd_skills if s in found], 'match')}</p>"
            f"<p><b>❌ Job skills not found</b><br>{chips([s for s in jd_skills if s not in found], 'missing')}</p>"
            f"<p><b>➕ Other skills</b><br>{chips([s for s in c['skills'] if s not in jd_skills], 'extra')}</p>"
            "</div>"
        )
        left, right = st.columns(2)
        left.markdown("**🎓 Education**")
        left.write(", ".join(c["education"]))
        right.markdown("**💼 Experience stated in resume**")
        right.write(", ".join(c["experience"]))
        with st.expander("Preview extracted resume text"):
            st.text(c["text"][:5000])

    with report:
        table = pd.DataFrame([
            {
                "Rank": rank,
                "Candidate Name": c["name"],
                "File Name": c["filename"],
                "Text Similarity (%)": c["score"],
                "Extracted Skills": ", ".join(c["skills"]),
                "Education": ", ".join(c["education"]),
                "Experience": ", ".join(c["experience"]),
            }
            for rank, c in enumerate(candidates, start=1)
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download ranking report (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="candidate_ranking_report.csv",
            mime="text/csv",
        )


show_html(STYLE)
show_html(
    '<div class="hero"><h1>📄 AI Resume Screening</h1>'
    "<p>Rank resumes by text similarity to a job description, using Sentence-BERT.</p></div>"
)

with st.sidebar:
    st.header("📤 Upload files")
    jd_file = st.file_uploader("📝 Job description (PDF or TXT)", type=["pdf", "txt"])
    jd_typed = st.text_area("Or paste the job description", height=150, disabled=jd_file is not None)
    resume_files = st.file_uploader(
        "👤 Candidate resumes (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Scanned image-only PDFs need OCR, which this app does not do.",
    )
    run = st.button("🚀 Screen candidates", type="primary", use_container_width=True)
    st.caption("Files are processed in memory and are not intentionally saved.")

# Results are kept across reruns, and cleared when any input changes
signature = hashlib.sha256()
for upload in ([jd_file] if jd_file else []) + list(resume_files or []):
    signature.update(upload.name.encode() + b"\0" + hashlib.sha256(upload.getvalue()).digest())
if not jd_file:
    signature.update(jd_typed.encode())
signature = signature.hexdigest()
if st.session_state.get("signature") != signature:
    st.session_state.pop("results", None)

if run:
    st.session_state.pop("results", None)
    job_description = read_upload(jd_file) if jd_file else jd_typed.strip()
    if not job_description:
        st.error("Upload or paste a readable job description in the sidebar.")
    elif not resume_files:
        st.warning("Upload at least one resume in the sidebar.")
    else:
        candidates = []
        for file in resume_files:
            text = read_upload(file)
            if text:
                candidates.append({
                    "name": Path(file.name).stem.replace("_", " ").replace("-", " ").title(),
                    "filename": file.name,
                    "text": text,
                    "skills": extract_skills(text),
                    "education": extract_education(text),
                    "experience": extract_experience(text),
                })
        if not candidates:
            st.error("No readable resumes were found. Try text-based PDF or TXT files.")
        else:
            try:
                with st.spinner("Comparing job description and resumes..."):
                    ranked = rank_candidates(job_description, candidates)
                st.session_state["results"] = (ranked, extract_skills(job_description))
                st.session_state["signature"] = signature
            except Exception as exc:
                st.error(f"Could not complete resume screening: {exc}")

if "results" in st.session_state:
    show_results(*st.session_state["results"])
elif not run:
    st.info("👈 Add a job description and resumes in the sidebar, then select **Screen candidates**.")

st.caption(
    "Scores are approximate text-similarity signals, not probabilities of job performance. "
    "Review each resume yourself and do not use scores as the sole hiring criterion."
)
