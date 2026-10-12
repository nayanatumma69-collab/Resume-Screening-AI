# 📄 AI Resume Screening & Candidate Ranking System

An automated AI-powered candidate evaluation and screening system that parses applicant resumes, extracts structured career attributes (skills, education, experience), and calculates semantic relevance against target job descriptions using Sentence-BERT.

🔗 **Live Streamlit Demo:** [resume-screening-ai-8g7xzgkhckndkfznumxlah.streamlit.app](https://resume-screening-ai-8g7xzgkhckndkfznumxlah.streamlit.app)

---

## 💡 How the Project Works

1. **Document Ingestion & Text Extraction:**
   - Supports both PDF and plain text formats.
   - Text is parsed page-by-page using `PyPDF2`, stripped of extraneous symbols/whitespace, and normalized for NLP evaluation.

2. **Attribute & Keyword Extraction:**
   - **Skills:** Matches candidate text against a targeted technology taxonomy using word-boundary regex patterns.
   - **Education:** Identifies academic credentials (e.g., B.Tech, M.Tech, Ph.D., MBA, BCA) using flexible regular expressions.
   - **Experience:** Detects years of relevant industry experience mentioned throughout the profile.

3. **Semantic Embedding & Matching:**
   - Converts the target Job Description and all candidate resumes into high-dimensional dense vector embeddings using Sentence-BERT (`all-MiniLM-L6-v2`).
   - Computes semantic alignment via **Cosine Similarity**, capturing contextual relevance rather than relying on naive keyword counts.
   - Scores are normalized to match percentages ($0\% \text{ to } 100\%$).

4. **Ranking & Reporting:**
   - Ranks candidates in descending order based on match scores.
   - Presents results via an interactive dashboard with match indicators and provides a one-click CSV export (`candidate_ranking_report.csv`).

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.x
- **Web Interface:** Streamlit
- **Document Parsing:** PyPDF2
- **NLP & Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Similarity & Math:** Scikit-Learn (`cosine_similarity`), NumPy
- **Data Manipulation & Export:** Pandas

---

## 🚀 How to Run the Project Locally

### 1. Clone the Repository
```bash
git clone <https://github.com/nayanatumma69-collab/Resume-Screening-AI.git>
cd AI_Resume_Screening
