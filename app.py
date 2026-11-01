from __future__ import annotations

import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from cvmatch import (
    fetch_jobs, infer_job_categories, load_resume_text, rank_tfidf, rank_embeddings,
    evaluate_rankings, save_jobs
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
JOBS_PATH = DATA_DIR / "jobs.json"

st.set_page_config(page_title="CVMatch POC", page_icon="🧩", layout="wide")

# Кастомна синьо-фіолетова (Blue & Violet AI) тема через CSS
st.markdown(
    """
    <style>
    /* Основний фон додатку — дуже м'який холодно-блакитний з легким лавандовим відтінком */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 50%, #F5F3FF 100%);
        color: #0F172A;
    }

    /* Сайдбар — глибокий темно-синій із фіолетовим відтінком */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1B4B 0%, #1E3A8A 100%);
        border-right: 1px solid #312E81;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] input {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    /* Головний заголовок з синьо-фіолетовим градієнтом */
    h1 {
        background: linear-gradient(90deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Підзаголовки секцій */
    h2, h3 {
        color: #1E1B4B !important;
        font-weight: 700 !important;
    }

    /* Основні кнопки (Primary) — яскравий градієнт Blue -> Violet */
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #2563EB 0%, #6366F1 50%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
        filter: brightness(1.06);
    }
    div.stButton > button:disabled {
        background: #CBD5E1 !important;
        color: #64748B !important;
        box-shadow: none !important;
    }

    /* Кнопки завантаження CSV (Download button) */
    div.stDownloadButton > button {
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
        border: 1.5px solid #6366F1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    div.stDownloadButton > button:hover {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }

    /* Картки Expander, форми та завантажувач файлів */
    div[data-testid="stExpander"],
    div[data-testid="stForm"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #C7D2FE !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.06) !important;
    }

    /* Поля вводу тексту */
    textarea, input[type="text"] {
        border-radius: 10px !important;
        border: 1px solid #A5B4FC !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Таблиці даних */
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.05);
    }

    /* Посилання */
    a {
        color: #4F46E5 !important;
        font-weight: 600;
        text-decoration: none;
    }
    a:hover {
        color: #7C3AED !important;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CVMatch — Proof of Concept")
st.caption("Resume-to-job matching: TF-IDF baseline vs. Sentence Transformers")

with st.sidebar:
    st.header("Налаштування")
    top_k = st.number_input("Кількість результатів", min_value=1, max_value=20, value=5)
    st.markdown("---")

st.subheader("1. Резюме")
resume_file = st.file_uploader("Завантажте резюме (.txt або .pdf)", type=["txt", "pdf"])
default_resume = """Data Engineer with experience in Python, SQL, Apache Spark and Databricks.
Built ETL/ELT pipelines, worked with AWS S3, Glue, data warehouses and relational databases.
Experienced in data quality checks, batch processing, distributed data processing"""
resume_text = default_resume
if resume_file:
    try:
        resume_text = load_resume_text(resume_file.getvalue(), resume_file.name)
        st.success(f"Текст резюме прочитано: {len(resume_text)} символів.")
    except Exception as exc:
        resume_text = ""
        st.error(f"Не вдалося прочитати файл: {exc}")

with st.expander("Переглянути / відредагувати текст резюме", expanded=not bool(resume_file)):
    resume_text = st.text_area("Текст резюме", value=resume_text, height=180)

st.subheader("2. Отримання вакансій")
search_categories = infer_job_categories(resume_text)
st.caption(
    "Пошук за напрямами, визначеними з резюме: "
    + ", ".join(search_categories)
    + "."
)
col_a, col_b = st.columns([1, 2])
with col_a:
    fetch_clicked = st.button(
        "Знайти вакансії за резюме",
        type="primary",
        use_container_width=True,
        disabled=not resume_text.strip(),
    )
with col_b:
    st.caption("DOU та Djinni шукаються за визначеними напрямами; результати кешуються локально у data/jobs.json.")

if fetch_clicked:
    with st.spinner("Шукаю актуальні вакансії на DOU та Djinni за вашим резюме..."):
        try:
            jobs, errors = fetch_jobs(categories=search_categories)
            if jobs:
                save_jobs(jobs, JOBS_PATH)
                st.success(f"Завантажено {len(jobs)} унікальних вакансій.")
            else:
                st.warning("Сайти не повернули вакансій за цими напрямами. Перевірте інтернет або завантажте попередній локальний набір.")
            for error in errors:
                st.warning(error)
        except Exception as exc:
            st.error(f"Помилка отримання вакансій: {exc}")

jobs = []
if JOBS_PATH.exists():
    try:
        jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.error("Файл data/jobs.json пошкоджений. Видаліть його та завантажте вакансії повторно.")

if not jobs:
    st.info("Поки немає вакансій. Натисніть «Завантажити актуальні вакансії».")
    st.stop()

jobs_df = pd.DataFrame(jobs)
st.write(f"У локальному наборі: **{len(jobs_df)} вакансій**.")
if "source" in jobs_df:
    st.dataframe(jobs_df[["title", "company", "location", "source", "url"]], use_container_width=True, hide_index=True)

st.subheader("3. Ранжування")
if st.button("Знайти вакансії для цього резюме", type="primary", disabled=not resume_text.strip()):
    with st.spinner("Обчислюю TF-IDF та семантичні embeddings. Перший запуск завантажить мовну модель..."):
        try:
            tfidf = rank_tfidf(resume_text, jobs, top_k=int(top_k))
            semantic = rank_embeddings(resume_text, jobs, top_k=int(top_k))
            st.session_state["tfidf_results"] = tfidf
            st.session_state["semantic_results"] = semantic
            st.session_state["ranked_at"] = datetime.now(timezone.utc).isoformat()
            st.session_state["resume_text"] = resume_text
        except Exception as exc:
            st.error(f"Помилка ранжування: {exc}")
            st.stop()

if "tfidf_results" in st.session_state:
    left, right = st.columns(2)
    with left:
        st.markdown("### TF-IDF + Cosine Similarity")
        st.dataframe(st.session_state["tfidf_results"], use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Sentence Transformers + Cosine Similarity")
        st.dataframe(st.session_state["semantic_results"], use_container_width=True, hide_index=True)

    st.caption("Score — косинусна схожість у відповідному векторному просторі. Це не ймовірність працевлаштування.")
    st.download_button(
        "Завантажити результати ранжування (CSV)",
        pd.concat([
            st.session_state["tfidf_results"].assign(method="TF-IDF"),
            st.session_state["semantic_results"].assign(method="Sentence Transformers")
        ]).to_csv(index=False).encode("utf-8-sig"),
        file_name="cvmatch_rankings.csv",
        mime="text/csv"
    )

    st.subheader("4. Оцінювання релевантності")
    st.write("Позначте кожну вакансію оцінкою: 0 — нерелевантна, 1 — частково релевантна, 2 — релевантна. Використовуйте однакові оцінки для обох методів.")
    all_top = {}
    for method, frame in [
        ("TF-IDF", st.session_state["tfidf_results"]),
        ("Sentence Transformers", st.session_state["semantic_results"])
    ]:
        for _, row in frame.iterrows():
            all_top[row["job_id"]] = row

    if "judgments" not in st.session_state:
        st.session_state["judgments"] = {}
    with st.form("relevance_form"):
        for job_id, row in all_top.items():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{row['title']}** — {row['company']}  \n[{row['source']}]({row['url']})")
            with c2:
                val = st.selectbox("Релевантність", [0, 1, 2], index=st.session_state["judgments"].get(job_id, 0),
                                   key=f"grade_{job_id}", help="0 — ні; 1 — частково; 2 — так")
            st.session_state["judgments"][job_id] = val
        submitted = st.form_submit_button("Обчислити метрики")
    if submitted:
        judgments = st.session_state["judgments"]
        metrics = evaluate_rankings(
            st.session_state["tfidf_results"], st.session_state["semantic_results"], judgments
        )
        st.markdown("### Метрики якості")
        st.dataframe(pd.DataFrame(metrics).T.rename_axis("Метод").reset_index(), use_container_width=True, hide_index=True)
        st.caption("Precision@5 рахує частку оцінених релевантних вакансій (оцінка ≥ 1). nDCG@5 враховує позицію та градацію релевантності 0/1/2.")
        st.download_button(
            "Завантажити оцінювання (CSV)",
            pd.DataFrame([{"job_id": k, "relevance": v} for k, v in judgments.items()]).to_csv(index=False).encode("utf-8-sig"),
            file_name="cvmatch_relevance_judgments.csv", mime="text/csv"
        )

st.markdown("---")
st.caption("POC для дослідження. Перевіряйте актуальність вакансії на сайті-джерелі перед використанням.")
