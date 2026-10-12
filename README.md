# 📄 AI Resume Screening System

## 🚀 Live Demo

**Try the application here:** [AI Resume Screening](https://resume-screening-ai-8g7xzgkhckndkfznumxlah.streamlit.app/)

## 📌 Project Overview

The AI Resume Screening System is a Python-based application that helps compare candidate resumes against a job description. It extracts text from PDF resumes, generates semantic embeddings, and calculates similarity scores to help organize resumes according to their relevance to a specified role.

The project demonstrates how Natural Language Processing (NLP), sentence embeddings, and semantic similarity can support an initial resume screening workflow.

This application is intended to assist recruiters and hiring teams, not replace human evaluation or hiring decisions.

## ⚙️ How the Project Works

1. **Job Description Input:** The user enters the job description containing the role's responsibilities, required qualifications, and relevant skills.
2. **Resume Upload:** The user uploads candidate resumes in PDF format through the Streamlit interface.
3. **Text Extraction:** PyPDF2 extracts readable text from each PDF resume.
4. **Semantic Embeddings:** The Sentence-Transformers model `all-MiniLM-L6-v2` converts the job description and resume text into numerical vector representations.
5. **Similarity Calculation:** Cosine similarity compares the resume embeddings with the job description embedding.
6. **Candidate Ranking:** The application sorts resumes by their calculated similarity scores to help identify potentially relevant profiles.
7. **Information Extraction:** The application uses heuristic-based logic to identify selected skills, education details, and experience-related information from resume text.
8. **Results and Export:** The Streamlit interface displays screening results and supports exporting results to a CSV file.

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Streamlit** – Web interface and interactive application
- **Sentence-Transformers** – Semantic text embeddings
- **all-MiniLM-L6-v2** – Pre-trained sentence embedding model
- **PyPDF2** – Text extraction from PDF resumes
- **Scikit-learn** – Cosine similarity and related machine learning utilities
- **NumPy** – Numerical operations
- **Pandas** – Data organization and CSV export

## ✨ Key Features

- Upload and process multiple PDF resumes
- Compare resumes against a job description
- Semantic similarity scoring using sentence embeddings
- Rank candidate resumes by similarity score
- Extract selected skills, education, and experience information using heuristic-based methods
- Display results through an interactive Streamlit interface
- Export screening results to CSV
- Browser-based access through Streamlit Community Cloud

## 💻 How to Run the Project Locally

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git (optional, for cloning the repository)

### Step 1: Clone the Repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ai-resume-screening.git
```

Navigate to the project directory:

```bash
cd ai-resume-screening
```

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

### Step 4: Open the Application

Streamlit will display a local URL in your terminal, usually:

```text
http://localhost:8501
```

Open the URL in your browser, enter a job description, upload PDF resumes, and review the screening results.

**Note:** The first run may take several minutes because the application needs to download and load the sentence embedding model. Available memory and processing resources may affect performance.

## 📂 Project Structure

```text
ai-resume-screening/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── Resume_Screening_AI.ipynb
```

- `app.py` – Main Streamlit application and resume screening logic
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation
- `notebooks/` – Original development notebook, if included

## ⚠️ Limitations and Responsible Use

- Similarity scores indicate textual or semantic similarity to the job description. They are not probabilities of job performance or measures of a candidate's overall suitability.
- Skills, education, and experience extraction uses heuristic-based logic and may miss information or produce inaccurate results.
- Scanned PDFs or image-based resumes may not yield readable text without OCR.
- Resume ranking should not be used as the sole basis for hiring or rejecting candidates.
- Recruiters should review candidates fairly and consider relevant experience, transferable skills, and the full context of each application.
- Resumes may contain personal and sensitive information. Use authorized documents, handle them securely, and avoid uploading confidential candidate data to public services without appropriate permission.

## 🔮 Future Improvements

- Add OCR support for scanned resumes.
- Improve structured extraction of skills, education, and work experience.
- Add configurable skill matching and job requirement filters.
- Support additional document formats.
- Integrate secure resume storage and access controls.
- Evaluate the system against human-reviewed screening results for accuracy and fairness.

## 🎯 Learning Outcomes

This project demonstrates practical skills in PDF text extraction, semantic embeddings, NLP, cosine similarity, candidate ranking, data processing, CSV export, and Streamlit application deployment.

## 👩‍💻 Author

Developed as an AI learning and portfolio project to demonstrate semantic similarity-based resume screening.
