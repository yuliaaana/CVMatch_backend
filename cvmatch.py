from __future__ import annotations
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Відповідність категорій між DOU та Djinni
CATEGORY_MAP: dict[str, dict[str, str]] = {
    "Усі категорії": {"dou": "", "djinni": ""},
    "Python": {"dou": "Python", "djinni": "python"},
    "Data Science / ML": {"dou": "Data Science", "djinni": "data-science"},
    "Data Engineer": {"dou": "Big Data", "djinni": "data-eng"},
    "Data Analyst": {"dou": "Analyst", "djinni": "data-analyst"},
    "JavaScript / Front-End": {"dou": "Front End", "djinni": "javascript"},
    "Java": {"dou": "Java", "djinni": "java"},
    ".NET / C#": {"dou": ".NET", "djinni": "dotnet"},
    "DevOps / SRE": {"dou": "DevOps", "djinni": "devops"},
    "QA / Automation": {"dou": "QA", "djinni": "qa"},
}

UKRAINIAN_STOP_WORDS = {
    "та", "і", "й", "у", "в", "на", "з", "із", "зі", "до", "по", "за", "від",
    "для", "про", "при", "або", "чи", "але", "проте", "однак", "як", "що", "щоб",
    "якщо", "коли", "де", "куди", "звідки", "це", "цей", "ця", "це", "ці", "той",
    "та", "те", "ті", "ми", "ви", "вони", "він", "вона", "воно", "наш", "ваш",
    "їх", "свій", "який", "яка", "яке", "які", "котрий", "бути", "є", "був",
    "була", "були", "буде", "має", "мати", "можна", "треба", "потрібно", "дуже",
    "також", "ще", "вже", "лише", "тільки", "навіть", "між", "над", "під", "перед",
    "після", "через", "без", "більше", "менше", "робота", "роботи", "компанія",
    "компанії", "команда", "команді", "шукаємо", "пропонуємо", "вимоги", "обов'язки",
    "умови", "досвід", "рівень", "знання", "розуміння", "навички", "перевагою",
    "буде", "плюсом", "рік", "років", "роки", "кандидат", "вакансія",
}

BILINGUAL_STOP_WORDS = list(ENGLISH_STOP_WORDS.union(UKRAINIAN_STOP_WORDS))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    soup = BeautifulSoup(str(value), "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_dou_title(raw_title: str) -> tuple[str, str, str]:
    """
    Типовий заголовок у RSS DOU має формат:
    'Senior Python Developer в CompanyName, Київ, віддалено'
    або 'Data Engineer в CompanyName, Львів'
    """
    title = raw_title.strip()
    company = "Не вказано"
    location = "Україна / Remote"

    if " в " in title:
        pos_part, rest = title.split(" в ", 1)
        title = pos_part.strip()
    elif " at " in title:
        pos_part, rest = title.split(" at ", 1)
        title = pos_part.strip()
    else:
        rest = ""

    if rest:
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if parts:
            company = parts[0]
        if len(parts) > 1:
            location = ", ".join(parts[1:])

    return title, company, location


def _format_pub_date(raw_date: str) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return clean_text(raw_date)


def _make_job_id(source: str, url: str) -> str:
    canonical = url.split("?")[0].rstrip("/")
    short_hash = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{source.lower()}:{short_hash}"


def fetch_dou_jobs(category: str = "", timeout: int = 20) -> list[dict]:
    """Отримує вакансії з офіційної RSS-стрічки DOU.ua."""
    url = "https://jobs.dou.ua/vacancies/feeds/"
    params = {"category": category} if category else {}

    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    jobs: list[dict] = []

    for item in root.findall("./channel/item"):
        raw_title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        description = clean_text(item.findtext("description"))
        pub_date = _format_pub_date(clean_text(item.findtext("pubDate")))

        if not raw_title or not link:
            continue

        title, company, location = _parse_dou_title(raw_title)
        canonical_url = link.split("?")[0].rstrip("/")
        searchable = " ".join(x for x in [title, company, location, category, description] if x)

        jobs.append({
            "job_id": _make_job_id("dou", canonical_url),
            "title": title,
            "company": company,
            "location": location,
            "description": description,
            "tags": category or "General IT",
            "published": pub_date,
            "source": "DOU",
            "url": canonical_url,
            "text": searchable[:20000],
        })

    return jobs


def fetch_djinni_jobs(keyword: str = "", timeout: int = 20) -> list[dict]:
    """
    Отримує вакансії з RSS-стрічки Djinni.co (з fallback-парсингом публічної HTML-сторінки,
    якщо RSS тимчасово закритий або змінений).
    """
    rss_url = "https://djinni.co/jobs/rss/"
    params = {"primary_keyword": keyword} if keyword else {}

    jobs: list[dict] = []
    response = requests.get(rss_url, params=params, headers=HEADERS, timeout=timeout)

    # Спроба 1: парсинг офіційного RSS Djinni
    if response.status_code == 200 and b"<rss" in response.content[:500]:
        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item"):
            title = clean_text(item.findtext("title"))
            link = clean_text(item.findtext("link"))
            description = clean_text(item.findtext("description"))
            pub_date = _format_pub_date(clean_text(item.findtext("pubDate")))
            categories = [clean_text(c.text) for c in item.findall("category") if c.text]

            if not title or not link:
                continue

            canonical_url = link.split("?")[0].rstrip("/")
            tags = ", ".join(categories) if categories else (keyword or "IT")
            searchable = " ".join(x for x in [title, tags, description] if x)

            jobs.append({
                "job_id": _make_job_id("djinni", canonical_url),
                "title": title,
                "company": "Djinni (Анонімно / Компанія)",
                "location": "Україна / Remote",
                "description": description,
                "tags": tags,
                "published": pub_date,
                "source": "Djinni",
                "url": canonical_url,
                "text": searchable[:20000],
            })
        if jobs:
            return jobs

    # Спроба 2 (Fallback): парсинг публічного каталогу https://djinni.co/jobs/
    html_url = "https://djinni.co/jobs/"
    html_resp = requests.get(html_url, params=params, headers=HEADERS, timeout=timeout)
    html_resp.raise_for_status()

    soup = BeautifulSoup(html_resp.text, "html.parser")
    for item in soup.select("li.list-jobs__item, div.job-item"):
        title_el = item.select_one("a.job-item__title-link, a.h3, a[href*='/jobs/']")
        if not title_el:
            continue
        title = clean_text(title_el.get_text())
        href = title_el.get("href", "")
        if not href:
            continue
        full_url = f"https://djinni.co{href}" if href.startswith("/") else href
        canonical_url = full_url.split("?")[0].rstrip("/")

        comp_el = item.select_one("a.mr-2, span.company-name")
        company = clean_text(comp_el.get_text()) if comp_el else "Djinni Employer"

        desc_el = item.select_one("div.job-list-item__description, span.js-original-text")
        description = clean_text(desc_el.get_text()) if desc_el else clean_text(item.get_text())

        loc_el = item.select_one("span.location-text")
        location = clean_text(loc_el.get_text()) if loc_el else "Україна / Remote"

        searchable = " ".join(x for x in [title, company, location, keyword, description] if x)
        jobs.append({
            "job_id": _make_job_id("djinni", canonical_url),
            "title": title,
            "company": company,
            "location": location,
            "description": description,
            "tags": keyword or "IT",
            "published": "",
            "source": "Djinni",
            "url": canonical_url,
            "text": searchable[:20000],
        })

    return jobs


def fetch_jobs(
    categories: list[str] | None = None,
    sources: tuple[str, ...] = ("DOU", "Djinni"),
    timeout: int = 20,
) -> tuple[list[dict], list[str]]:
    """
    Завантажує вакансії з DOU та Djinni за обраними категоріями.
    Повертає список унікальних вакансій та список помилок (якщо були).
    """
    if not categories:
        categories = ["Усі категорії"]

    jobs: list[dict] = []
    errors: list[str] = []

    for cat_label in categories:
        mapping = CATEGORY_MAP.get(cat_label, {"dou": "", "djinni": ""})

        if "DOU" in sources:
            try:
                dou_records = fetch_dou_jobs(category=mapping["dou"], timeout=timeout)
                jobs.extend(dou_records)
            except Exception as exc:
                errors.append(f"DOU ({cat_label}): {type(exc).__name__}: {exc}")

        if "Djinni" in sources:
            try:
                djinni_records = fetch_djinni_jobs(keyword=mapping["djinni"], timeout=timeout)
                jobs.extend(djinni_records)
            except Exception as exc:
                errors.append(f"Djinni ({cat_label}): {type(exc).__name__}: {exc}")

    # Дедуплікація за канонічним URL
    unique: dict[str, dict] = {}
    for job in jobs:
        key = job["url"].rstrip("/")
        if key not in unique:
            unique[key] = job

    return list(unique.values()), errors


def load_resume_text(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    raise ValueError("Підтримуються лише PDF та TXT.")


def _job_frame(jobs: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(jobs)
    if frame.empty:
        raise ValueError("Немає вакансій для ранжування. Спочатку завантажте вакансії.")
    frame["text"] = frame["text"].fillna("")
    frame = frame[frame["text"].str.strip().ne("")].copy()
    if frame.empty:
        raise ValueError("У вакансій немає тексту для аналізу.")
    return frame


def _result_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    result["score"] = scores
    result = result.sort_values("score", ascending=False).head(top_k)
    result["score"] = result["score"].round(4)
    cols = ["job_id", "title", "company", "location", "source", "score"]
    if extra_cols:
        cols.extend(c for c in extra_cols if c in result.columns)
    cols.append("url")
    return result[cols].reset_index(drop=True)


def rank_tfidf(resume_text: str, jobs: list[dict], top_k: int = 5) -> pd.DataFrame:
    frame = _job_frame(jobs)
    corpus = [resume_text] + frame["text"].tolist()

    # Кастомний token_pattern зберігає C++, C#, .NET, Node.js, українські та латинські літери
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=BILINGUAL_STOP_WORDS,
        token_pattern=r"(?u)\b[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9_+#.]{2,}\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    resume_vec = matrix[0:1]
    job_vecs = matrix[1:]

    scores = cosine_similarity(resume_vec, job_vecs).ravel()

    # Визначаємо топ спільних термінів між резюме та кожною вакансією для інтерпретованості
    feature_names = np.array(vectorizer.get_feature_names_out())
    resume_weights = resume_vec.toarray().ravel()

    matched_terms_list = []
    for idx in range(job_vecs.shape[0]):
        job_weights = job_vecs[idx].toarray().ravel()
        overlap = resume_weights * job_weights
        top_indices = np.argsort(overlap)[::-1][:5]
        terms = [feature_names[i] for i in top_indices if overlap[i] > 0]
        matched_terms_list.append(", ".join(terms))

    frame["matched_terms"] = matched_terms_list
    return _result_frame(frame, scores, top_k, extra_cols=["matched_terms"])


def rank_embeddings(
    resume_text: str,
    jobs: list[dict],
    top_k: int = 5,
    model: Any = None,
) -> pd.DataFrame:
    frame = _job_frame(jobs)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    query_embedding = model.encode(
        [resume_text], normalize_embeddings=True, show_progress_bar=False
    )
    job_embeddings = model.encode(
        frame["text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    scores = np.dot(job_embeddings, query_embedding[0])
    return _result_frame(frame, scores, top_k)


def dcg(relevances: list[int]) -> float:
    return sum(
        (2**rel - 1) / np.log2(position + 2)
        for position, rel in enumerate(relevances)
    )


def reciprocal_rank(relevances: list[int]) -> float:
    for position, rel in enumerate(relevances, start=1):
        if rel >= 1:
            return 1.0 / position
    return 0.0


def evaluate_rankings(
    tfidf_results: pd.DataFrame,
    semantic_results: pd.DataFrame,
    judgments: dict[str, int],
) -> dict[str, dict[str, float]]:
    output = {}
    k = max(len(tfidf_results), len(semantic_results), 1)

    for name, frame in [
        ("TF-IDF", tfidf_results),
        ("Sentence Transformers", semantic_results),
    ]:
        rels = [int(judgments.get(job_id, 0)) for job_id in frame["job_id"].tolist()]
        n = max(len(rels), 1)
        precision = sum(rel >= 1 for rel in rels) / n
        ideal = sorted((int(v) for v in judgments.values()), reverse=True)[: len(rels)]
        ideal_dcg = dcg(ideal)
        ndcg = dcg(rels) / ideal_dcg if ideal_dcg else 0.0
        mrr = reciprocal_rank(rels)

        output[name] = {
            f"Precision@{k}": round(precision, 4),
            f"nDCG@{k}": round(ndcg, 4),
            f"MRR@{k}": round(mrr, 4),
            f"Relevant in Top-{k}": int(sum(rel >= 1 for rel in rels)),
        }
    return output


def save_jobs(jobs: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
