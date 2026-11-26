import html
import io
import re
from pathlib import Path

import altair as alt
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

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #4f46e5, #9333ea 50%, #db2777);
        border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
    }
    .hero h1 { color: #fff; margin: 0; padding: 0; font-size: 2.1rem; }
    .hero p { color: #f5f3ff; margin: 8px 0 0; }
    .card {
        border-radius: 14px; padding: 16px 18px; color: #fff;
        margin-bottom: 12px; min-height: 108px;
    }
    .card small { display: block; opacity: .9; }
    .card b { display: block; font-size: 1.6rem; overflow-wrap: anywhere; }
    .chip {
        display: inline-block; padding: 3px 11px; margin: 3px;
        border-radius: 999px; font-size: .85rem; color: #fff;
    }
    .chip-match { background: #047857; }
    .chip-missing { background: #be123c; }
    .chip-extra { background: #4f46e5; }
    div.stButton > button[kind="primary"], div.stDownloadButton > button {
        background: linear-gradient(90deg, #4f46e5, #db2777);
        color: #fff; border: 0; border-radius: 12px; font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(147, 51, 234, .45); border-radius: 12px;
    }
    section[data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, rgba(79,70,229,.16), rgba(219,39,119,.10));
    }
    </style>
    <div class="hero">
        <h1>📄 AI Resume Screening & Candidate Ranking</h1>
        <p>Upload resumes and compare their text with a job description using Sentence-BERT.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
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

def stat_card(column, label, value, colors):
    column.markdown(
        f'<div class="card" style="background:linear-gradient(135deg,{colors})">'
        f"<small>{label}</small><b>{html.escape(str(value))}</b></div>",
        unsafe_allow_html=True,
    )

def skill_chips(skills, kind):
    if not skills:
        return "<em>None detected</em>"
    return "".join(
        f'<span class="chip chip-{kind}">{html.escape(skill)}</span>' for skill in skills
    )

def match_chart(candidates):
    """Horizontal bar chart of text similarity, best candidate on top."""
    data = pd.DataFrame(
        {"Candidate": f"#{rank} {c['name']}", "Similarity": c["score"]}
        for rank, c in enumerate(candidates, start=1)
    )
    base = alt.Chart(data).encode(
        y=alt.Y("Candidate:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("Similarity:Q", scale=alt.Scale(domain=[0, 100]), title="Text similarity (%)"),
        tooltip=["Candidate", alt.Tooltip("Similarity:Q", format=".2f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=6).encode(
        color=alt.Color(
            "Similarity:Q",
            scale=alt.Scale(domain=[0, 100], range=["#f59e0b", "#db2777", "#4f46e5"]),
            legend=None,
        )
    )
    labels = base.mark_text(align="left", dx=5, color="#8b5cf6", fontWeight="bold").encode(
        text=alt.Text("Similarity:Q", format=".1f")
    )
    return (bars + labels).properties(height=max(140, 46 * len(data)))

def skill_chart(candidates, jd_skills):
    """Stacked bar chart of job-description skills found and not found."""
    rows = []
    for rank, c in enumerate(candidates, start=1):
        matched = len(set(c["skills"]) & set(jd_skills))
        label = f"#{rank} {c['name']}"
        rows.append({"Candidate": label, "Status": "Found", "Skills": matched, "Order": 0})
        rows.append({"Candidate": label, "Status": "Not found",
                     "Skills": len(jd_skills) - matched, "Order": 1})
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("Candidate:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Skills:Q", title="Job description skills"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Found", "Not found"], range=["#10b981", "#f43f5e"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            order=alt.Order("Order:Q"),
            tooltip=["Candidate", "Status", "Skills"],
        )
        .properties(height=max(160, 46 * len(candidates)))
    )

with st.sidebar:
    st.header("✨ How it works")
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
    st.subheader("📝 1. Job description")
    job_description = st.text_area(
        "Paste the target job description",
        value=DEFAULT_JD,
        height=300,
    )
with col_upload:
    st.subheader("📂 2. Upload candidate resumes")
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
                jd_skills = extract_skills(job_description)
                top = parsed_candidates[0]
                average = sum(c["score"] for c in parsed_candidates) / len(parsed_candidates)

                k1, k2, k3, k4 = st.columns(4)
                stat_card(k1, "👥 Candidates screened", len(parsed_candidates), "#4f46e5,#6366f1")
                stat_card(k2, "🏆 Top ranked", top["name"], "#7e22ce,#a855f7")
                stat_card(k3, "🎯 Highest similarity", f"{top['score']:.1f}%", "#be185d,#ec4899")
                stat_card(k4, "📈 Average similarity", f"{average:.1f}%", "#0f766e,#0d9488")

                st.subheader("📊 Candidate match graph")
                st.altair_chart(match_chart(parsed_candidates), use_container_width=True)

                if jd_skills:
                    st.subheader("🧩 Skill coverage")
                    st.altair_chart(skill_chart(parsed_candidates, jd_skills), use_container_width=True)
                    st.caption(
                        "Keyword check against the app's skill list. It does not affect the ranking."
                    )

                st.subheader("🏅 Candidate ranking")
                for rank, candidate in enumerate(parsed_candidates, start=1):
                    with st.expander(
                        f"{({1: '🥇', 2: '🥈', 3: '🥉'}).get(rank, '🔹')} #{rank} — {candidate['name']} · {candidate['score']:.2f}% text similarity",
                        expanded=(rank == 1),
                    ):
                        st.progress(candidate["score"] / 100.0)
                        c1, c2, c3 = st.columns(3)
                        found = set(candidate["skills"])
                        c1.markdown("**✅ Skills matching the job**")
                        c1.markdown(skill_chips(sorted(found & set(jd_skills)), "match"), unsafe_allow_html=True)
                        c1.markdown("**❌ Job skills not found**")
                        c1.markdown(skill_chips(sorted(set(jd_skills) - found), "missing"), unsafe_allow_html=True)
                        c1.markdown("**➕ Other skills**")
                        c1.markdown(skill_chips(sorted(found - set(jd_skills)), "extra"), unsafe_allow_html=True)
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
