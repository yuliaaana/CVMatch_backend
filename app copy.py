from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from cvmatch import (
    CATEGORY_MAP,
    evaluate_rankings,
    fetch_jobs,
    load_resume_text,
    rank_embeddings,
    rank_tfidf,
    save_jobs,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
JOBS_PATH = DATA_DIR / "jobs.json"

st.set_page_config(page_title="CVMatch", page_icon="💚", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --emerald: #10b981;
        --emerald-dark: #047857;
        --emerald-hover: #065f46;
        --emerald-border: rgba(16, 185, 129, 0.38);
    }

    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--text-color);
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2.2rem, 5vw, 3.2rem) !important;
        font-weight: 750 !important;
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid var(--emerald-border);
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--emerald);
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 0.7rem;
        border: 1px solid var(--emerald-dark);
        background: var(--emerald-dark);
        color: white;
        font-weight: 650;
        transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--emerald-hover);
        background: var(--emerald-hover);
        color: white;
        transform: translateY(-1px);
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stMultiSelect"] [role="combobox"] {
        border-radius: 0.65rem;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed var(--emerald-border);
        border-radius: 0.9rem;
        background: var(--secondary-background-color);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--emerald-border);
        border-radius: 0.85rem;
        overflow: hidden;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--text-color);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("CVMatch")
st.caption("Знайдіть вакансії, що відповідають вашому досвіду.")


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


with st.sidebar:
    st.header("Пошук")
    top_k = st.number_input("Кількість результатів", min_value=1, max_value=20, value=5)

    selected_sources = st.multiselect(
        "Джерела",
        options=["DOU", "Djinni"],
        default=["DOU", "Djinni"],
    )
    selected_categories = st.multiselect(
        "Категорії вакансій",
        options=list(CATEGORY_MAP.keys()),
        default=["Python", "Data Science / ML", "Data Engineer"],
    )

st.subheader("Резюме")
resume_file = st.file_uploader("Завантажте резюме", type=["txt", "pdf"])
default_resume = """Data Engineer / Python Developer з досвідом розробки ETL/ELT пайплайнів.
Стек: Python, SQL, Apache Spark, PySpark, Airflow, Databricks, Docker, Git.
Працював з хмарними сервісами AWS (S3, Glue, Redshift, Lambda), реляційними БД (PostgreSQL) та аналітичними сховищами даних.
Досвід побудови перевірок якості даних (data quality), пакетної обробки та оптимізації SQL-запитів."""

resume_text = default_resume
if resume_file:
    try:
        resume_text = load_resume_text(resume_file.getvalue(), resume_file.name)
        st.success(f"Текст резюме прочитано: {len(resume_text)} символів.")
    except Exception as exc:
        st.error(f"Не вдалося прочитати файл: {exc}")

with st.expander("Переглянути або змінити текст резюме", expanded=not bool(resume_file)):
    resume_text = st.text_area("Текст резюме", value=resume_text, height=180)

st.subheader("Вакансії")
col_a, _ = st.columns([1, 2])
with col_a:
    fetch_clicked = st.button(
        "Оновити вакансії",
        type="primary",
        use_container_width=True,
        disabled=not selected_sources,
    )

if fetch_clicked:
    with st.spinner("Оновлюю вакансії..."):
        try:
            jobs, errors = fetch_jobs(
                categories=selected_categories or ["Усі категорії"],
                sources=tuple(selected_sources),
            )
            if jobs:
                save_jobs(jobs, JOBS_PATH)
                st.success(f"Завантажено та збережено {len(jobs)} унікальних вакансій.")
            else:
                st.warning("Не вдалося отримати вакансії. Перевірте з'єднання або змініть категорії.")
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
    st.info("Вакансій поки немає. Натисніть «Оновити вакансії».")
    st.stop()

jobs_df = pd.DataFrame(jobs)

# Швидка фільтрація локальної бази за джерелом
filter_col1, _ = st.columns([2, 2])
with filter_col1:
    available_sources = jobs_df["source"].unique().tolist() if "source" in jobs_df else []
    active_sources = st.multiselect(
        "Фільтр локальної бази за джерелом:",
        options=available_sources,
        default=available_sources,
    )

filtered_jobs = [j for j in jobs if j.get("source") in active_sources]
filtered_df = pd.DataFrame(filtered_jobs)

st.write(f"У локальному наборі доступно для пошуку: **{len(filtered_df)} вакансій** (всього в кеші: {len(jobs_df)}).")
if not filtered_df.empty:
    display_cols = [c for c in ["title", "company", "location", "tags", "source", "published", "url"] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("Посилання")},
    )

st.subheader("Результати")
if st.button("Знайти відповідні вакансії", type="primary", disabled=not resume_text.strip() or not filtered_jobs):
    with st.spinner("Обчислюю TF-IDF та семантичні embeddings..."):
        try:
            model = get_embedding_model()
            tfidf = rank_tfidf(resume_text, filtered_jobs, top_k=int(top_k))
            semantic = rank_embeddings(resume_text, filtered_jobs, top_k=int(top_k), model=model)
            st.session_state["tfidf_results"] = tfidf
            st.session_state["semantic_results"] = semantic
            st.session_state["ranked_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            st.error(f"Помилка ранжування: {exc}")
            st.stop()

if "tfidf_results" in st.session_state:
    left, right = st.columns(2)
    with left:
        st.markdown("### TF-IDF")
        st.dataframe(
            st.session_state["tfidf_results"],
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("URL")},
        )
    with right:
        st.markdown("### Семантичний пошук")
        st.dataframe(
            st.session_state["semantic_results"],
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("URL")},
        )

    st.download_button(
        "Завантажити результати CSV",
        pd.concat([
            st.session_state["tfidf_results"].assign(method="TF-IDF"),
            st.session_state["semantic_results"].assign(method="Sentence Transformers"),
        ]).to_csv(index=False).encode("utf-8-sig"),
        file_name="cvmatch_ua_rankings.csv",
        mime="text/csv",
    )

    st.subheader("Оцінка релевантності")

    all_top = {}
    for frame in [st.session_state["tfidf_results"], st.session_state["semantic_results"]]:
        for _, row in frame.iterrows():
            all_top[row["job_id"]] = row

    # Мапа повних описів для зручної перевірки під час оцінювання
    desc_map = {j["job_id"]: j.get("description", "") for j in jobs}

    if "judgments" not in st.session_state:
        st.session_state["judgments"] = {}

    with st.form("relevance_form"):
        for job_id, row in all_top.items():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"**{row['title']}** — {row['company']} ({row['location']})  \n"
                    f"Джерело: [{row['source']}]({row['url']})"
                )
                snippet = desc_map.get(job_id, "")[:350]
                if snippet:
                    st.caption(f"{snippet}...")
            with c2:
                val = st.selectbox(
                    "Релевантність",
                    options=[0, 1, 2],
                    index=st.session_state["judgments"].get(job_id, 0),
                    key=f"grade_{job_id}",
                    help="0 — нерелевантна; 1 — частково; 2 — релевантна",
                )
            st.session_state["judgments"][job_id] = val
            st.divider()

        submitted = st.form_submit_button("Обчислити метрики якості")

    if submitted:
        judgments = st.session_state["judgments"]
        metrics = evaluate_rankings(
            st.session_state["tfidf_results"],
            st.session_state["semantic_results"],
            judgments,
        )
        st.markdown("### Метрики якості")
        metrics_df = pd.DataFrame(metrics).T.rename_axis("Метод").reset_index()
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Завантажити оцінки CSV",
            pd.DataFrame([{"job_id": k, "relevance": v} for k, v in judgments.items()])
            .to_csv(index=False)
            .encode("utf-8-sig"),
            file_name="cvmatch_ua_relevance_judgments.csv",
            mime="text/csv",
        )
