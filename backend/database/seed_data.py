from __future__ import annotations 

import sqlite3
from datetime import datetime

from backend.config import settings

DB_PATH = str(settings.sqlite_path)

CANDIDATES = [
    {
        "name": "Arjun Sharma",
        "email": "arjun.sharma@example.com",
        "phone": "+91-9876543210",
        "location": "Bangalore, India",
        "experience_years": 4,
        "education": "B.Tech Computer Science, IIT Bombay",
        "summary": "Full-stack AI engineer specializing in LLM applications, RAG pipelines, and FastAPI backends.",
        "skills": [
            ("Python", "Expert", 4),
            ("FastAPI", "Expert", 3),
            ("LangChain", "Advanced", 2),
            ("Machine Learning", "Advanced", 3),
            ("PostgreSQL", "Intermediate", 3),
            ("Docker", "Intermediate", 2),
        ],
        "projects": [
            ("AI Document Copilot", "RAG-based document QA system using LangChain and ChromaDB", "Python, FastAPI, LangChain, ChromaDB, Streamlit", "github.com/arjun/doc-copilot"),
            ("ML Pipeline Orchestrator", "Automated ML pipeline with feature engineering and model registry", "Python, MLflow, Scikit-learn, Airflow", "github.com/arjun/ml-pipeline"),
        ],
    },
    {
        "name": "Priya Nair",
        "email": "priya.nair@example.com",
        "phone": "+91-9123456789",
        "location": "Hyderabad, India",
        "experience_years": 3,
        "education": "M.Tech AI, IIIT Hyderabad",
        "summary": "Machine learning engineer focused on NLP, transformer models, and model deployment.",
        "skills": [
            ("Python", "Expert", 3),
            ("Machine Learning", "Expert", 3),
            ("NLP", "Advanced", 2),
            ("PyTorch", "Advanced", 2),
            ("HuggingFace", "Advanced", 2),
            ("FastAPI", "Intermediate", 1),
        ],
        "projects": [
            ("Sentiment Analysis API", "Production NLP API for real-time sentiment classification", "Python, FastAPI, HuggingFace, Docker", "github.com/priya/sentiment-api"),
            ("Resume Parser", "Automated resume parsing using spaCy and transformer models", "Python, spaCy, HuggingFace, Flask", "github.com/priya/resume-parser"),
        ],
    },
    {
        "name": "Rahul Verma",
        "email": "rahul.verma@example.com",
        "phone": "+91-9988776655",
        "location": "Pune, India",
        "experience_years": 5,
        "education": "B.E. Information Technology, VIT Pune",
        "summary": "Backend engineer with deep expertise in Python microservices, API design, and cloud deployment.",
        "skills": [
            ("Python", "Expert", 5),
            ("FastAPI", "Expert", 4),
            ("Django", "Advanced", 4),
            ("PostgreSQL", "Expert", 5),
            ("AWS", "Advanced", 3),
            ("Docker", "Expert", 3),
            ("Kubernetes", "Intermediate", 2),
        ],
        "projects": [
            ("E-commerce Microservices", "Scalable microservices platform handling 1M+ daily transactions", "Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes", "github.com/rahul/ecom-microservices"),
            ("Real-time Analytics Dashboard", "Live analytics pipeline with WebSocket streaming", "Python, FastAPI, Redis, ClickHouse, React", "github.com/rahul/analytics-dash"),
        ],
    },
    {
        "name": "Sneha Patel",
        "email": "sneha.patel@example.com",
        "phone": "+91-9765432109",
        "location": "Ahmedabad, India",
        "experience_years": 2,
        "education": "B.Tech Data Science, NIT Surat",
        "summary": "Data scientist with expertise in statistical modeling, data visualization, and ML experimentation.",
        "skills": [
            ("Python", "Advanced", 2),
            ("Machine Learning", "Advanced", 2),
            ("Data Analysis", "Expert", 2),
            ("SQL", "Advanced", 2),
            ("TensorFlow", "Intermediate", 1),
            ("Tableau", "Advanced", 2),
        ],
        "projects": [
            ("Customer Churn Predictor", "ML model to predict churn with 91% accuracy for a telecom company", "Python, Scikit-learn, XGBoost, Streamlit", "github.com/sneha/churn-predictor"),
            ("Stock Price Forecaster", "LSTM-based time series forecasting for equities", "Python, TensorFlow, Pandas, Plotly", "github.com/sneha/stock-forecast"),
        ],
    },
    {
        "name": "Vikram Singh",
        "email": "vikram.singh@example.com",
        "phone": "+91-9654321098",
        "location": "Delhi, India",
        "experience_years": 6,
        "education": "M.S. Computer Science, Delhi University",
        "summary": "Senior ML engineer and team lead with experience in production AI systems and LLMOps.",
        "skills": [
            ("Python", "Expert", 6),
            ("Machine Learning", "Expert", 6),
            ("LangChain", "Expert", 2),
            ("LLMOps", "Advanced", 2),
            ("FastAPI", "Advanced", 3),
            ("Kubernetes", "Advanced", 3),
            ("MLflow", "Expert", 3),
        ],
        "projects": [
            ("LLM Evaluation Framework", "Automated testing and evaluation suite for LLM outputs", "Python, LangChain, FastAPI, MLflow, PostgreSQL", "github.com/vikram/llm-eval"),
            ("Hybrid Search Engine", "BM25 + vector search hybrid for enterprise document retrieval", "Python, FastAPI, ChromaDB, Elasticsearch", "github.com/vikram/hybrid-search"),
        ],
    },
    {
        "name": "Ananya Reddy",
        "email": "ananya.reddy@example.com",
        "phone": "+91-9543210987",
        "location": "Chennai, India",
        "experience_years": 3,
        "education": "B.Tech CSE, Anna University",
        "summary": "Full-stack developer with React frontend expertise and Python backend skills.",
        "skills": [
            ("Python", "Advanced", 3),
            ("React", "Expert", 3),
            ("JavaScript", "Expert", 3),
            ("FastAPI", "Intermediate", 1),
            ("MongoDB", "Advanced", 2),
            ("TypeScript", "Advanced", 2),
        ],
        "projects": [
            ("AI Chat Interface", "Real-time chat UI with streaming LLM responses", "React, TypeScript, FastAPI, WebSockets", "github.com/ananya/ai-chat"),
            ("Portfolio Builder", "Drag-and-drop portfolio creation tool with AI content suggestions", "React, Node.js, OpenAI API, MongoDB", "github.com/ananya/portfolio-builder"),
        ],
    },
    {
        "name": "Karthik Iyer",
        "email": "karthik.iyer@example.com",
        "phone": "+91-9432109876",
        "location": "Bangalore, India",
        "experience_years": 4,
        "education": "B.Tech ECE, NITK Surathkal",
        "summary": "DevOps and MLOps engineer with strong Python scripting and cloud infrastructure skills.",
        "skills": [
            ("Python", "Advanced", 4),
            ("Docker", "Expert", 4),
            ("Kubernetes", "Expert", 3),
            ("AWS", "Expert", 4),
            ("CI/CD", "Expert", 4),
            ("Terraform", "Advanced", 2),
            ("Machine Learning", "Intermediate", 1),
        ],
        "projects": [
            ("ML Serving Platform", "Kubernetes-native ML model serving with auto-scaling", "Python, Kubernetes, Docker, FastAPI, Prometheus", "github.com/karthik/ml-serving"),
            ("Infrastructure as Code Toolkit", "Reusable Terraform modules for AWS ML workloads", "Terraform, AWS, Python, GitHub Actions", "github.com/karthik/iac-toolkit"),
        ],
    },
    {
        "name": "Meera Joshi",
        "email": "meera.joshi@example.com",
        "phone": "+91-9321098765",
        "location": "Mumbai, India",
        "experience_years": 1,
        "education": "B.Sc Computer Science, Mumbai University",
        "summary": "Junior developer eager to grow in AI/ML. Strong Python fundamentals and active open-source contributor.",
        "skills": [
            ("Python", "Intermediate", 1),
            ("Machine Learning", "Beginner", 1),
            ("SQL", "Intermediate", 1),
            ("JavaScript", "Intermediate", 1),
            ("Git", "Advanced", 1),
        ],
        "projects": [
            ("News Summarizer Bot", "Telegram bot that summarizes news articles using transformers", "Python, HuggingFace, Telegram API", "github.com/meera/news-bot"),
        ],
    },
    {
        "name": "Rohan Gupta",
        "email": "rohan.gupta@example.com",
        "phone": "+91-9210987654",
        "location": "Kolkata, India",
        "experience_years": 5,
        "education": "B.Tech IT, Jadavpur University",
        "summary": "Data engineer specializing in large-scale data pipelines, SQL optimization, and real-time streaming.",
        "skills": [
            ("Python", "Expert", 5),
            ("SQL", "Expert", 5),
            ("Apache Spark", "Advanced", 3),
            ("Kafka", "Advanced", 3),
            ("PostgreSQL", "Expert", 5),
            ("dbt", "Advanced", 2),
            ("Airflow", "Advanced", 3),
        ],
        "projects": [
            ("Real-time Data Lake", "Kafka + Spark streaming pipeline ingesting 10M events/day", "Python, Kafka, Apache Spark, Delta Lake, Airflow", "github.com/rohan/data-lake"),
            ("SQL Analytics Engine", "Custom query engine with automatic index recommendations", "Python, PostgreSQL, FastAPI, React", "github.com/rohan/sql-engine"),
        ],
    },
    {
        "name": "Divya Menon",
        "email": "divya.menon@example.com",
        "phone": "+91-9109876543",
        "location": "Kochi, India",
        "experience_years": 3,
        "education": "M.Sc Data Science, University of Kerala",
        "summary": "AI researcher with a focus on computer vision, generative models, and explainable AI.",
        "skills": [
            ("Python", "Expert", 3),
            ("Machine Learning", "Expert", 3),
            ("Computer Vision", "Advanced", 2),
            ("PyTorch", "Expert", 3),
            ("Generative AI", "Advanced", 2),
            ("FastAPI", "Intermediate", 1),
        ],
        "projects": [
            ("Defect Detection System", "Real-time CV pipeline for manufacturing quality control", "Python, PyTorch, FastAPI, OpenCV, Docker", "github.com/divya/defect-detect"),
            ("Explainable ML Dashboard", "Interactive SHAP/LIME visualizations for model interpretability", "Python, Streamlit, SHAP, Scikit-learn", "github.com/divya/xai-dashboard"),
        ],
    },
]


def seed() -> None:
    """Insert all candidates, skills, and projects if table is empty."""
    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if existing > 0:
            print(f"[seed_data] Skipping — {existing} candidates already in DB.")
            return

        now = datetime.utcnow().isoformat()
        for c in CANDIDATES:
            cur = conn.execute(
                """INSERT INTO candidates
                   (name, email, phone, location, experience_years, education, summary, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    c["name"], c["email"], c["phone"], c["location"],
                    c["experience_years"], c["education"], c["summary"], now,
                ),
            )
            cid = cur.lastrowid

            for skill_name, proficiency, years in c["skills"]:
                conn.execute(
                    "INSERT INTO skills (candidate_id, skill_name, proficiency, years) VALUES (?,?,?,?)",
                    (cid, skill_name, proficiency, years),
                )

            for title, desc, tech, url in c.get("projects", []):
                conn.execute(
                    "INSERT INTO projects (candidate_id, title, description, tech_stack, github_url) VALUES (?,?,?,?,?)",
                    (cid, title, desc, tech, url),
                )

        conn.commit()
        print(f"[seed_data] Seeded {len(CANDIDATES)} candidates with skills and projects.")


if __name__ == "__main__":
    from backend.database.db_manager import init_db
    init_db()
    seed()
    print("Done.")
