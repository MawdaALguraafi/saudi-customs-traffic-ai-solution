import os
import re
import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(APP_DIR, "saudi_customs_faiss")
FORECAST_PATH = os.path.join(APP_DIR, "customs_forecast.csv")


st.set_page_config(
    page_title="Saudi Customs Traffic AI Solution",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.html(
    """
    <style>
    :root {
        --background: #06110d;
        --sidebar: #081912;
        --card: #0d2119;
        --card-light: #123126;
        --card-hover: #173d2e;
        --border: #245c43;
        --green: #22c55e;
        --green-light: #4ade80;
        --green-dark: #15803d;
        --white: #ffffff;
        --muted: #b7cabf;
    }

    html,
    body,
    .stApp {
        background-color: var(--background);
        color: var(--white);
    }

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(34, 197, 94, 0.08),
                transparent 35%
            ),
            var(--background);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    p,
    label,
    li {
        color: var(--white) !important;
    }

    .main-title {
        color: var(--white);
        font-size: 40px;
        font-weight: 800;
        line-height: 1.2;
        margin: 8px 0 6px 0;
    }

    .subtitle {
        color: var(--muted);
        font-size: 16px;
        margin-bottom: 26px;
    }

    .section-title {
        color: var(--green-light);
        font-size: 25px;
        font-weight: 750;
        margin: 12px 0 14px 0;
    }

    .info-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-left: 5px solid var(--green);
        border-radius: 14px;
        padding: 18px;
        color: var(--white);
        margin-bottom: 18px;
    }

    .info-card strong {
        color: var(--green-light) !important;
    }

    [data-testid="stMetric"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        min-height: 125px;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--green-light) !important;
        font-weight: 800;
    }

    [data-testid="stMetricDelta"] {
        color: var(--white) !important;
    }

    .stSelectbox label,
    .stTextInput label,
    .stMultiSelect label,
    .stDateInput label {
        color: var(--white) !important;
        font-weight: 650;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--white) !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--white) !important;
        -webkit-text-fill-color: var(--white) !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: var(--card-light) !important;
    }

    [role="option"] {
        background-color: var(--card-light) !important;
        color: var(--white) !important;
    }

    [role="option"] * {
        color: var(--white) !important;
    }

    [role="option"]:hover {
        background-color: var(--card-hover) !important;
    }

    input,
    textarea {
        background-color: var(--card) !important;
        color: var(--white) !important;
        -webkit-text-fill-color: var(--white) !important;
        border-color: var(--border) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #9fb7aa !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessage"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"] * {
        color: var(--white) !important;
    }

    /* Fully dark chat input, including Streamlit's outer wrapper */
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div {
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: none !important;
        padding: 8px 10px !important;
    }

    [data-testid="stChatInput"] div[data-baseweb="textarea"],
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {
        background-color: var(--card) !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: var(--card) !important;
        color: var(--white) !important;
        -webkit-text-fill-color: var(--white) !important;
        caret-color: var(--green-light) !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9fb7aa !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: var(--green-dark) !important;
        color: var(--white) !important;
        border: none !important;
        border-radius: 11px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #16a34a !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: var(--white) !important;
        color: var(--white) !important;
    }

    [data-testid="stExpander"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background-color: var(--card-light) !important;
        padding: 12px !important;
    }

    [data-testid="stExpander"] summary * {
        color: var(--green-light) !important;
        fill: var(--green-light) !important;
    }

    [data-testid="stExpanderDetails"] {
        background-color: var(--card) !important;
        color: var(--white) !important;
        padding: 18px;
    }

    [data-testid="stExpanderDetails"] * {
        color: var(--white) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px 10px 0 0;
        padding: 11px 24px;
    }

    .stTabs [data-baseweb="tab"] p {
        color: var(--muted) !important;
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--green-dark) !important;
        border-color: var(--green) !important;
    }

    .stTabs [aria-selected="true"] p {
        color: var(--white) !important;
        font-weight: 750;
    }

    .stButton button {
        width: 100%;
        background-color: var(--green-dark);
        color: var(--white) !important;
        border: 1px solid var(--green);
        border-radius: 9px;
        font-weight: 700;
    }

    .stButton button:hover {
        background-color: #20a64e;
        border-color: var(--green-light);
    }

    .stButton button * {
        color: var(--white) !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    [data-testid="stAlert"] {
        background-color: var(--card-light) !important;
        color: var(--white) !important;
        border: 1px solid var(--border);
    }

    [data-testid="stAlert"] * {
        color: var(--white) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    .hero-card {
        background: linear-gradient(135deg, #0d2119 0%, #0a1b14 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 26px 28px;
        margin-bottom: 20px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
    }

    .hero-eyebrow {
        color: var(--green-light);
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .kpi-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px 17px;
        min-height: 92px;
        margin-bottom: 12px;
    }

    .kpi-label {
        color: var(--muted) !important;
        font-size: 13px;
        font-weight: 650;
    }

    .kpi-value {
        color: var(--white) !important;
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }

    .suggestion-title {
        color: var(--muted) !important;
        font-size: 13px;
        font-weight: 700;
        margin: 6px 0 8px;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #123126;
        margin-left: 12%;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #0d2119;
        margin-right: 8%;
    }

    [data-testid="stDownloadButton"] button {
        width: 100%;
        background-color: var(--card-light);
        color: var(--white) !important;
        border: 1px solid var(--border);
        border-radius: 9px;
        font-weight: 700;
    }

    </style>
    """
)


@st.cache_data
def load_forecast() -> pd.DataFrame:
    forecast_path = FORECAST_PATH

    if not os.path.exists(forecast_path):
        return pd.DataFrame()

    forecast_data = pd.read_csv(forecast_path)

    if "ds" not in forecast_data.columns:
        return pd.DataFrame()

    forecast_data["ds"] = pd.to_datetime(
        forecast_data["ds"],
        errors="coerce"
    )

    numeric_columns = [
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]

    for column in numeric_columns:
        if column in forecast_data.columns:
            forecast_data[column] = pd.to_numeric(
                forecast_data[column],
                errors="coerce"
            )

    if "yhat" not in forecast_data.columns:
        return pd.DataFrame()

    forecast_data = forecast_data.dropna(
        subset=["ds", "yhat"]
    )

    forecast_data = forecast_data[
        forecast_data["ds"].between(
            "2026-01-01",
            "2026-12-31"
        )
    ].copy()

    for column in numeric_columns:
        if column in forecast_data.columns:
            forecast_data[column] = forecast_data[
                column
            ].clip(lower=0)

    return forecast_data.sort_values("ds").reset_index(
        drop=True
    )


@st.cache_resource
def load_vector_database() -> FAISS:
    embedding_model = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    vector_database = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_database


def create_language_model(
    api_key: str
) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
        max_retries=1,
        timeout=30
    )


def extract_text_content(
    response_content: Any
) -> str:
    if isinstance(response_content, str):
        return response_content.strip()

    if isinstance(response_content, list):
        text_parts = []

        for item in response_content:
            if isinstance(item, dict):
                text_value = item.get("text")

                if text_value:
                    text_parts.append(str(text_value))

            elif isinstance(item, str):
                text_parts.append(item)

            elif hasattr(item, "text"):
                text_value = getattr(item, "text", "")

                if text_value:
                    text_parts.append(str(text_value))

        final_text = "\n".join(
            part.strip()
            for part in text_parts
            if part and part.strip()
        )

        if final_text:
            return final_text

    if isinstance(response_content, dict):
        text_value = response_content.get("text")

        if text_value:
            return str(text_value).strip()

    return str(response_content).strip()



ARABIC_MONTHS = {
    "يناير": "01",
    "فبراير": "02",
    "مارس": "03",
    "أبريل": "04",
    "ابريل": "04",
    "مايو": "05",
    "يونيو": "06",
    "يوليو": "07",
    "أغسطس": "08",
    "اغسطس": "08",
    "سبتمبر": "09",
    "أكتوبر": "10",
    "اكتوبر": "10",
    "نوفمبر": "11",
    "ديسمبر": "12",
}

ENGLISH_MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

KNOWN_PORTS = [
    "البطحاء",
    "جسر الملك فهد",
    "سلوى",
    "الخفجي",
    "الربع الخالي",
    "الحديثة",
    "الرقعي",
    "الوديعة",
    "جديدة عرعر",
    "حالة عمار",
    "الدره",
]


def normalize_arabic_text(text: str) -> str:
    normalized = str(text).strip().lower()

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old_character, new_character in replacements.items():
        normalized = normalized.replace(
            old_character,
            new_character
        )

    normalized = re.sub(
        r"[ًٌٍَُِّْـ]",
        "",
        normalized
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    return normalized


def is_arabic_question(question: str) -> bool:
    return bool(
        re.search(
            r"[\u0600-\u06FF]",
            question
        )
    )


def extract_year_month(
    question: str
) -> str | None:
    question_lower = question.lower()

    direct_match = re.search(
        r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b",
        question_lower
    )

    if direct_match:
        year = direct_match.group(1)
        month = direct_match.group(2).zfill(2)
        return f"{year}-{month}"

    year_match = re.search(
        r"\b(20\d{2})\b",
        question_lower
    )

    if not year_match:
        return None

    year = year_match.group(1)

    normalized_question = normalize_arabic_text(
        question
    )

    for month_name, month_number in ARABIC_MONTHS.items():
        if normalize_arabic_text(month_name) in normalized_question:
            return f"{year}-{month_number}"

    for month_name, month_number in ENGLISH_MONTHS.items():
        if month_name in question_lower:
            return f"{year}-{month_number}"

    return None


def extract_ports(
    question: str
) -> list[str]:
    normalized_question = normalize_arabic_text(
        question
    )

    matched_ports = []

    for port in KNOWN_PORTS:
        if normalize_arabic_text(port) in normalized_question:
            matched_ports.append(port)

    return matched_ports


def detect_document_type(
    question: str
) -> str | None:
    normalized_question = normalize_arabic_text(
        question
    )

    anomaly_terms = [
        "anomaly",
        "anomalies",
        "شذوذ",
        "غير طبيعي",
        "ارتفاع مفاجي",
        "انخفاض مفاجي",
    ]

    if any(
        term in normalized_question
        for term in anomaly_terms
    ):
        return "anomaly"

    if extract_year_month(question):
        return "monthly_port_summary"

    specific_date_patterns = [
        r"\b20\d{2}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/20\d{2}\b",
    ]

    if any(
        re.search(pattern, question)
        for pattern in specific_date_patterns
    ):
        return "daily"

    return None



def extract_specific_date(
    question: str
) -> str | None:
    question_lower = question.lower()

    iso_match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        question_lower
    )

    if iso_match:
        year, month, day = iso_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    slash_match = re.search(
        r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b",
        question_lower
    )

    if slash_match:
        day, month, year = slash_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    year_match = re.search(
        r"\b(20\d{2})\b",
        question_lower
    )

    if not year_match:
        return None

    year = year_match.group(1)
    normalized_question = normalize_arabic_text(question)
    month_number = None

    for month_name, number in ARABIC_MONTHS.items():
        if normalize_arabic_text(month_name) in normalized_question:
            month_number = number
            break

    if month_number is None:
        for month_name, number in ENGLISH_MONTHS.items():
            if month_name in question_lower:
                month_number = number
                break

    if month_number is None:
        return None

    month_names = list(ARABIC_MONTHS) + list(ENGLISH_MONTHS)

    for month_name in month_names:
        match = re.search(
            rf"\b([0-3]?\d)\s+{re.escape(month_name)}\s+(20\d{{2}})\b",
            question_lower,
            flags=re.IGNORECASE
        )

        if match:
            day = int(match.group(1))
            matched_year = match.group(2)
            return f"{matched_year}-{month_number}-{day:02d}"

    return None


def is_anomaly_question(
    question: str
) -> bool:
    normalized_question = normalize_arabic_text(
        question
    )

    anomaly_terms = [
        "anomaly",
        "anomalies",
        "شذوذ",
        "غير طبيعي",
        "ارتفاع مفاجي",
        "انخفاض مفاجي",
    ]

    return any(
        term in normalized_question
        for term in anomaly_terms
    )


def document_matches_specific_date(
    document,
    requested_date: str
) -> bool:
    metadata = document.metadata or {}

    possible_metadata_fields = [
        "date",
        "ds",
        "anomaly_date",
        "record_date",
        "day",
    ]

    for field_name in possible_metadata_fields:
        value = metadata.get(field_name)

        if value is not None:
            parsed_value = pd.to_datetime(
                value,
                errors="coerce"
            )

            if (
                pd.notna(parsed_value)
                and parsed_value.strftime("%Y-%m-%d") == requested_date
            ):
                return True

    page_content = str(
        document.page_content
    )

    requested_timestamp = pd.Timestamp(
        requested_date
    )

    date_variants = [
        requested_date,
        requested_timestamp.strftime("%d/%m/%Y"),
        requested_timestamp.strftime("%Y/%m/%d"),
        requested_timestamp.strftime("%d %B %Y"),
    ]

    try:
        date_variants.append(
            requested_timestamp.strftime("%-d %B %Y")
        )
    except ValueError:
        pass

    return any(
        date_variant.lower() in page_content.lower()
        for date_variant in date_variants
    )


def retrieve_anomaly_documents(
    question: str,
    vector_database: FAISS
) -> list:
    requested_date = extract_specific_date(
        question
    )

    all_documents = get_all_vector_documents(
        vector_database
    )

    anomaly_documents = [
        document
        for document in all_documents
        if "anomaly" in str(
            (document.metadata or {}).get(
                "document_type",
                ""
            )
        ).lower()
    ]

    if requested_date:
        exact_date_documents = [
            document
            for document in anomaly_documents
            if document_matches_specific_date(
                document,
                requested_date
            )
        ]

        if exact_date_documents:
            return exact_date_documents

    retriever = vector_database.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8
        }
    )

    semantic_documents = retriever.invoke(
        question
    )

    semantic_anomalies = [
        document
        for document in semantic_documents
        if "anomaly" in str(
            (document.metadata or {}).get(
                "document_type",
                ""
            )
        ).lower()
    ]

    if requested_date:
        exact_semantic_documents = [
            document
            for document in semantic_anomalies
            if document_matches_specific_date(
                document,
                requested_date
            )
        ]

        if exact_semantic_documents:
            return exact_semantic_documents

    return semantic_anomalies[:3]

def get_all_vector_documents(
    vector_database: FAISS
) -> list:
    docstore = getattr(
        vector_database,
        "docstore",
        None
    )

    document_dictionary = getattr(
        docstore,
        "_dict",
        {}
    )

    return list(
        document_dictionary.values()
    )


def metadata_matches_document_type(
    metadata: dict,
    requested_type: str | None
) -> bool:
    if not requested_type:
        return True

    actual_type = str(
        metadata.get(
            "document_type",
            ""
        )
    ).lower()

    if requested_type == "monthly_port_summary":
        return actual_type == "monthly_port_summary"

    if requested_type == "anomaly":
        return "anomaly" in actual_type

    if requested_type == "daily":
        return "daily" in actual_type

    return True


def filter_documents_by_metadata(
    documents: list,
    year_month: str | None,
    ports: list[str],
    requested_type: str | None
) -> list:
    filtered_documents = []

    normalized_requested_ports = {
        normalize_arabic_text(port)
        for port in ports
    }

    for document in documents:
        metadata = document.metadata or {}

        if not metadata_matches_document_type(
            metadata,
            requested_type
        ):
            continue

        if year_month:
            document_year_month = str(
                metadata.get(
                    "year_month",
                    ""
                )
            )

            if document_year_month != year_month:
                continue

        if ports:
            document_port = normalize_arabic_text(
                metadata.get(
                    "port",
                    ""
                )
            )

            if document_port not in normalized_requested_ports:
                continue

        filtered_documents.append(document)

    return filtered_documents


def retrieve_customs_documents(
    question: str,
    vector_database: FAISS
) -> list:
    if is_anomaly_question(question):
        return retrieve_anomaly_documents(
            question=question,
            vector_database=vector_database
        )

    year_month = extract_year_month(
        question
    )

    ports = extract_ports(
        question
    )

    requested_type = detect_document_type(
        question
    )

    all_documents = get_all_vector_documents(
        vector_database
    )

    exact_documents = filter_documents_by_metadata(
        documents=all_documents,
        year_month=year_month,
        ports=ports,
        requested_type=requested_type
    )

    if exact_documents:
        return sorted(
            exact_documents,
            key=lambda document: (
                str(
                    (document.metadata or {}).get(
                        "port",
                        ""
                    )
                ),
                str(
                    (document.metadata or {}).get(
                        "year_month",
                        ""
                    )
                )
            )
        )

    retriever = vector_database.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8
        }
    )

    semantic_documents = retriever.invoke(
        question
    )

    filtered_semantic_documents = (
        filter_documents_by_metadata(
            documents=semantic_documents,
            year_month=year_month,
            ports=ports,
            requested_type=requested_type
        )
    )

    if filtered_semantic_documents:
        return filtered_semantic_documents

    return semantic_documents[:6]


def is_comparison_question(
    question: str
) -> bool:
    normalized_question = normalize_arabic_text(
        question
    )

    comparison_terms = [
        "compare",
        "comparison",
        "مقارنه",
        "قارن",
        "الفرق",
        "مقابل",
        "vs",
    ]

    return any(
        term in normalized_question
        for term in comparison_terms
    )


def is_highest_question(
    question: str
) -> bool:
    normalized_question = normalize_arabic_text(
        question
    )

    terms = [
        "highest",
        "maximum",
        "most traffic",
        "اعلي",
        "الاكثر",
        "اكبر",
    ]

    return any(
        term in normalized_question
        for term in terms
    )


def is_lowest_question(
    question: str
) -> bool:
    normalized_question = normalize_arabic_text(
        question
    )

    terms = [
        "lowest",
        "minimum",
        "least traffic",
        "اقل",
        "الاقل",
        "اصغر",
    ]

    return any(
        term in normalized_question
        for term in terms
    )


def format_monthly_record(
    document,
    arabic: bool
) -> str:
    metadata = document.metadata or {}

    port = metadata.get(
        "port",
        "Unknown"
    )

    year_month = metadata.get(
        "year_month",
        "Unknown"
    )

    total_movements = float(
        metadata.get(
            "total_movements",
            0
        )
    )

    peak_hour = metadata.get(
        "peak_hour"
    )

    dominant_direction = metadata.get(
        "dominant_direction",
        "Unknown"
    )

    if peak_hour is not None:
        peak_hour_text = f"{int(peak_hour):02d}:00"
    else:
        peak_hour_text = (
            "غير متوفر"
            if arabic
            else "Unavailable"
        )

    if arabic:
        return (
            f"**منفذ {port} ({year_month})**\n\n"
            f"- إجمالي الحركة: **{total_movements:,.0f} حركة**.\n"
            f"- ساعة الذروة: **{peak_hour_text}**.\n"
            f"- الاتجاه السائد: **{dominant_direction}**."
        )

    return (
        f"**{port} port ({year_month})**\n\n"
        f"- Total traffic: **{total_movements:,.0f} movements**.\n"
        f"- Peak hour: **{peak_hour_text}**.\n"
        f"- Dominant direction: **{dominant_direction}**."
    )


def format_comparison_answer(
    question: str,
    documents: list
) -> str | None:
    requested_ports = extract_ports(
        question
    )

    year_month = extract_year_month(
        question
    )

    arabic = is_arabic_question(
        question
    )

    monthly_documents = [
        document
        for document in documents
        if (
            document.metadata or {}
        ).get(
            "document_type"
        ) == "monthly_port_summary"
    ]

    documents_by_port = {
        normalize_arabic_text(
            (document.metadata or {}).get(
                "port",
                ""
            )
        ): document
        for document in monthly_documents
    }

    selected_documents = []

    for port in requested_ports:
        matching_document = documents_by_port.get(
            normalize_arabic_text(port)
        )

        if matching_document:
            selected_documents.append(
                matching_document
            )

    if len(selected_documents) < 2:
        return None

    selected_documents = selected_documents[:2]

    first_document = selected_documents[0]
    second_document = selected_documents[1]

    first_metadata = first_document.metadata or {}
    second_metadata = second_document.metadata or {}

    first_total = float(
        first_metadata.get(
            "total_movements",
            0
        )
    )

    second_total = float(
        second_metadata.get(
            "total_movements",
            0
        )
    )

    difference = abs(
        first_total - second_total
    )

    higher_document = (
        first_document
        if first_total >= second_total
        else second_document
    )

    higher_port = (
        higher_document.metadata or {}
    ).get(
        "port",
        "Unknown"
    )

    record_texts = [
        format_monthly_record(
            document,
            arabic
        )
        for document in selected_documents
    ]

    if arabic:
        conclusion = (
            f"**الخلاصة:** كان منفذ **{higher_port}** الأعلى حركة "
            f"بفارق **{difference:,.0f} حركة**"
        )

        if year_month:
            conclusion += f" خلال **{year_month}**."

        return "\n\n".join(
            record_texts + [conclusion]
        )

    conclusion = (
        f"**Conclusion:** **{higher_port}** recorded more traffic "
        f"by **{difference:,.0f} movements**"
    )

    if year_month:
        conclusion += f" during **{year_month}**."

    return "\n\n".join(
        record_texts + [conclusion]
    )


def format_ranking_answer(
    question: str,
    documents: list
) -> str | None:
    arabic = is_arabic_question(
        question
    )

    monthly_documents = [
        document
        for document in documents
        if (
            document.metadata or {}
        ).get(
            "document_type"
        ) == "monthly_port_summary"
        and (
            document.metadata or {}
        ).get(
            "total_movements"
        ) is not None
    ]

    if not monthly_documents:
        return None

    if is_highest_question(question):
        selected_document = max(
            monthly_documents,
            key=lambda document: float(
                (
                    document.metadata or {}
                ).get(
                    "total_movements",
                    0
                )
            )
        )

        ranking_word = (
            "الأعلى"
            if arabic
            else "highest"
        )

    elif is_lowest_question(question):
        selected_document = min(
            monthly_documents,
            key=lambda document: float(
                (
                    document.metadata or {}
                ).get(
                    "total_movements",
                    0
                )
            )
        )

        ranking_word = (
            "الأقل"
            if arabic
            else "lowest"
        )

    else:
        return None

    metadata = selected_document.metadata or {}

    port = metadata.get(
        "port",
        "Unknown"
    )

    total_movements = float(
        metadata.get(
            "total_movements",
            0
        )
    )

    year_month = metadata.get(
        "year_month",
        "Unknown"
    )

    if arabic:
        return (
            f"منفذ **{port}** هو {ranking_word} حركة خلال "
            f"**{year_month}**، بإجمالي **{total_movements:,.0f} حركة**."
        )

    return (
        f"**{port}** had the {ranking_word} traffic during "
        f"**{year_month}**, with **{total_movements:,.0f} movements**."
    )



def format_anomaly_fallback(
    question: str,
    documents: list
) -> str | None:
    anomaly_documents = [
        document
        for document in documents
        if "anomaly" in str(
            (document.metadata or {}).get("document_type", "")
        ).lower()
    ]

    if not anomaly_documents:
        return None

    document = anomaly_documents[0]
    metadata = document.metadata or {}
    arabic = is_arabic_question(question)

    date_value = (
        metadata.get("date")
        or metadata.get("anomaly_date")
        or metadata.get("ds")
        or extract_specific_date(question)
        or "Unknown"
    )

    actual_value = (
        metadata.get("actual_movements")
        or metadata.get("actual")
        or metadata.get("total_movements")
    )

    expected_value = (
        metadata.get("expected_movements")
        or metadata.get("expected")
        or metadata.get("rolling_mean")
    )

    change_value = (
        metadata.get("change_percentage")
        or metadata.get("percentage_change")
        or metadata.get("deviation_percent")
    )

    details = []

    if actual_value is not None:
        try:
            details.append(
                ("الحركة الفعلية", f"{float(actual_value):,.0f}")
                if arabic
                else ("Actual traffic", f"{float(actual_value):,.0f}")
            )
        except (TypeError, ValueError):
            pass

    if expected_value is not None:
        try:
            details.append(
                ("الحركة المتوقعة", f"{float(expected_value):,.0f}")
                if arabic
                else ("Expected traffic", f"{float(expected_value):,.0f}")
            )
        except (TypeError, ValueError):
            pass

    if change_value is not None:
        try:
            details.append(
                ("نسبة التغير", f"{float(change_value):,.1f}%")
                if arabic
                else ("Change", f"{float(change_value):,.1f}%")
            )
        except (TypeError, ValueError):
            pass

    if details:
        if arabic:
            lines = [f"في **{date_value}** تم رصد حركة غير طبيعية:"]
        else:
            lines = [f"On **{date_value}**, an unusual traffic pattern was detected:"]

        lines.extend(
            f"- {label}: **{value}**"
            for label, value in details
        )

        return "\n".join(lines)

    # The anomaly document itself is already grounded in the indexed data.
    content = str(document.page_content).strip()

    if content:
        return content

    return None


def generate_local_fallback_answer(
    question: str,
    vector_database: FAISS
):
    documents = retrieve_customs_documents(
        question=question,
        vector_database=vector_database
    )

    documents = documents[:4]

    if not documents:
        answer = (
            "لم يتم العثور على سجلات مناسبة للسؤال."
            if is_arabic_question(question)
            else "No relevant records were found for this question."
        )
        return answer, []

    if is_comparison_question(question):
        answer = format_comparison_answer(
            question,
            documents
        )
        if answer:
            return answer, documents

    if (
        is_highest_question(question)
        or is_lowest_question(question)
    ):
        answer = format_ranking_answer(
            question,
            documents
        )
        if answer:
            return answer, documents

    if is_anomaly_question(question):
        answer = format_anomaly_fallback(
            question,
            documents
        )
        if answer:
            return answer, documents

    # Final grounded fallback for other questions.
    top_document = documents[0]
    content = str(top_document.page_content).strip()

    if content:
        return content, documents

    answer = (
        "تم استرجاع السجل، لكن لا توجد تفاصيل نصية كافية للإجابة."
        if is_arabic_question(question)
        else "A record was retrieved, but it does not contain enough text to answer."
    )

    return answer, documents


def invoke_model_with_retry(
    language_model: ChatGoogleGenerativeAI,
    messages: list,
    maximum_attempts: int = 2
):
    last_error = None

    for attempt_number in range(
        1,
        maximum_attempts + 1
    ):
        try:
            return language_model.invoke(
                messages
            )

        except Exception as error:
            last_error = error
            error_text = str(error)

            quota_error = (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
            )

            if (
                not quota_error
                or attempt_number == maximum_attempts
            ):
                raise

            retry_match = re.search(
                r"retry(?:Delay| in)?['\": ]+(\d+)",
                error_text,
                flags=re.IGNORECASE
            )

            wait_seconds = (
                min(int(retry_match.group(1)) + 1, 12)
                if retry_match
                else 8
            )

            time.sleep(wait_seconds)

    raise last_error


def ask_customs_rag(
    question: str,
    vector_database: FAISS,
    language_model: ChatGoogleGenerativeAI
):
    documents = retrieve_customs_documents(
        question=question,
        vector_database=vector_database
    )

    # Send only the most relevant records to Gemini to reduce
    # prompt size and response time.
    documents = documents[:4]

    if not documents:
        no_data_answer = (
            "لم يتم العثور على سجلات مناسبة للسؤال ضمن بيانات الـRAG."
            if is_arabic_question(question)
            else "No relevant records were found in the RAG data."
        )

        return no_data_answer, []

    context_parts = []

    for document_number, document in enumerate(
        documents,
        start=1
    ):
        metadata = document.metadata or {}

        context_parts.append(
            f"""
Retrieved record {document_number}

Content:
{document.page_content}

Metadata:
{metadata}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    requested_year_month = extract_year_month(
        question
    )

    requested_date = extract_specific_date(
        question
    )

    requested_ports = extract_ports(
        question
    )

    system_prompt = """
You are an AI assistant specialized in Saudi customs port
traffic analysis.

You must generate the final answer using only the retrieved
customs traffic records provided in the prompt.

Rules:
1. Every final answer must be written by you as the language model.
2. Use monthly aggregated records for monthly questions.
3. Use daily records for specific-date questions.
4. Use anomaly records for anomaly questions.
5. For comparison questions, compare only the ports explicitly
   requested by the user.
6. For highest or lowest traffic questions, rank only records from
   the requested month.
7. Clearly state port names, dates, totals, peak hours and dominant
   directions when those values are available.
8. Observation dates are sampled and are not continuous daily data.
9. Do not invent, estimate or infer missing values.
10. If a requested record was not retrieved, say that it was not
    retrieved in the current context. Never claim that it does not
    exist in the complete dataset.
11. Respond in the same language used by the user.
12. Keep the answer very concise, clear and structured. Use no more than 120 words.
13. Do not mention metadata, embeddings, vector databases, retrieval
    mechanics or internal technical details.
14. Return only the final answer as normal text.
"""

    user_prompt = f"""
Retrieved customs traffic records:

{context}

Detected request details:
- Requested month: {requested_year_month}
- Requested date: {requested_date}
- Requested ports: {requested_ports}

User question:

{question}
"""

    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_prompt
        )
    ]

    response = invoke_model_with_retry(
        language_model=language_model,
        messages=messages
    )

    answer = extract_text_content(
        response.content
    )

    return answer, documents

def style_figure(
    figure: go.Figure,
    title: str
) -> go.Figure:
    figure.update_layout(
        title={
            "text": title,
            "font": {
                "size": 20,
                "color": "#ffffff"
            }
        },
        template="plotly_dark",
        paper_bgcolor="#06110d",
        plot_bgcolor="#0d2119",
        font={
            "color": "#ffffff",
            "size": 13
        },
        margin={
            "l": 35,
            "r": 30,
            "t": 65,
            "b": 40
        },
        legend={
            "font": {
                "color": "#ffffff"
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        hovermode="x unified"
    )

    figure.update_xaxes(
        gridcolor="#245c43",
        linecolor="#245c43",
        tickfont={
            "color": "#ffffff"
        },
        title_font={
            "color": "#ffffff"
        }
    )

    figure.update_yaxes(
        gridcolor="#245c43",
        linecolor="#245c43",
        tickfont={
            "color": "#ffffff"
        },
        title_font={
            "color": "#ffffff"
        }
    )

    return figure


def display_retrieved_evidence(
    documents
) -> None:
    with st.expander(
        "Retrieved RAG Evidence"
    ):
        for document_number, document in enumerate(
            documents,
            start=1
        ):
            st.markdown(
                f"**Retrieved record {document_number}**"
            )

            st.write(document.page_content)

            if document.metadata:
                st.caption(
                    f"Metadata: {document.metadata}"
                )

            if document_number < len(documents):
                st.divider()


forecast = load_forecast()


api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or ""
).strip()

st.html(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">AI Decision Support</div>
        <div class="main-title">Saudi Customs Traffic AI</div>
        <div class="subtitle" style="margin-bottom: 0;">
            Explore customs traffic insights through an AI assistant
            and review the full-year 2026 traffic forecast.
        </div>
    </div>
    """
)


rag_tab, forecast_tab = st.tabs(
    [
        "AI Assistant",
        "Traffic Forecast"
    ]
)


with rag_tab:
    st.html(
        """
        <div class="section-title">Customs Traffic AI Assistant</div>
        <div class="info-card">
            <strong>Ask questions in Arabic or English.</strong><br><br>
            Explore ports, monthly traffic, peak periods and detected
            anomalies using answers grounded in retrieved customs records.
        </div>
        """
    )

    kpi_1, kpi_2, kpi_3 = st.columns(3)

    with kpi_1:
        st.html(
            '<div class="kpi-card"><div class="kpi-label">LAND PORTS</div>'
            '<div class="kpi-value">11</div></div>'
        )

    with kpi_2:
        st.html(
            '<div class="kpi-card"><div class="kpi-label">TRAFFIC RECORDS</div>'
            '<div class="kpi-value">72K+</div></div>'
        )

    with kpi_3:
        st.html(
            '<div class="kpi-card"><div class="kpi-label">AI CAPABILITY</div>'
            '<div class="kpi-value">RAG + GEMINI</div></div>'
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    st.markdown(
        '<div class="suggestion-title">Suggested questions</div>',
        unsafe_allow_html=True
    )

    suggestion_1, suggestion_2, suggestion_3 = st.columns(3)

    with suggestion_1:
        if st.button(
            "Highest-traffic port",
            key="suggestion_highest",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Which customs port had the highest total traffic "
                "during January 2025?"
            )
            st.rerun()

    with suggestion_2:
        if st.button(
            "Compare two ports",
            key="suggestion_compare",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "Compare الربع الخالي and الدره during January 2025."
            )
            st.rerun()

    with suggestion_3:
        if st.button(
            "Explain an anomaly",
            key="suggestion_anomaly",
            use_container_width=True
        ):
            st.session_state.pending_question = (
                "What happened during the detected traffic anomaly "
                "on 5 January 2025?"
            )
            st.rerun()

    clear_col, spacer_col = st.columns([1, 3])

    with clear_col:
        if st.button(
            "Clear conversation",
            key="clear_chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

    # Process the question before rendering the conversation.
    # This keeps the chat input visually below all questions and answers.
    if st.session_state.pending_question:
        submitted_question = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

        st.session_state.messages.append(
            {
                "role": "user",
                "content": submitted_question
            }
        )

        if not api_key:
            answer = (
                "The Gemini API key is not configured. Add GOOGLE_API_KEY "
                "or GEMINI_API_KEY to the Colab environment, then restart the app."
            )

            retrieved_documents = []

        elif not os.path.exists(
            FAISS_PATH
        ):
            answer = (
                "The customs FAISS index was not found. "
                f"Expected folder: {FAISS_PATH}"
            )

            retrieved_documents = []

        else:
            try:
                with st.spinner(
                    "Retrieving customs records and generating the answer..."
                ):
                    vector_database = load_vector_database()
                    language_model = create_language_model(
                        api_key
                    )

                    answer, retrieved_documents = ask_customs_rag(
                        submitted_question,
                        vector_database,
                        language_model
                    )

            except Exception as error:
                # If Gemini is unavailable for any reason (quota, model,
                # authentication, timeout or service error), keep the app
                # answering from the retrieved FAISS records.
                try:
                    vector_database = load_vector_database()

                    answer, retrieved_documents = (
                        generate_local_fallback_answer(
                            submitted_question,
                            vector_database
                        )
                    )

                except Exception as fallback_error:
                    answer = (
                        "The AI assistant could not generate an answer."
                        f"\n\nGemini error: {error}"
                        f"\n\nFallback error: {fallback_error}"
                    )
                    retrieved_documents = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "documents": retrieved_documents
            }
        )

        st.rerun()

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! Ask me about Saudi customs ports, traffic patterns, "
                "monthly comparisons or detected anomalies."
            )

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("documents")
            ):
                display_retrieved_evidence(
                    message["documents"]
                )

    typed_question = st.chat_input(
        "Ask about customs traffic..."
    )

    if typed_question:
        cleaned_question = typed_question.strip()

        if cleaned_question:
            st.session_state.pending_question = (
                cleaned_question
            )

            st.rerun()


with forecast_tab:
    st.html(
        """
        <div class="section-title">
            2026 Customs Traffic Forecast
        </div>

        <div class="info-card">
            <strong>Planning view for decision makers.</strong>
            <br><br>
            Review monthly demand, daily trends and the expected peak period
            from 1 January through 31 December 2026.
        </div>
        """
    )

    if forecast.empty:
        st.error(
            "No valid 2026 forecast records were found in "
            "customs_forecast.csv."
        )

        st.code(
            """
Expected forecast period:
2026-01-01 to 2026-12-31

Expected rows:
365
            """
        )

    else:
        forecast_start = forecast["ds"].min()
        forecast_end = forecast["ds"].max()
        forecast_days = len(forecast)

        forecast_total = forecast["yhat"].sum()
        forecast_average = forecast["yhat"].mean()

        peak_row = forecast.loc[
            forecast["yhat"].idxmax()
        ]

        peak_value = peak_row["yhat"]
        peak_date = peak_row["ds"]

        metric_1, metric_2, metric_3 = st.columns(3)

        metric_1.metric(
            "2026 Forecast Total",
            f"{forecast_total:,.0f}"
        )

        metric_2.metric(
            "Average Daily Traffic",
            f"{forecast_average:,.0f}"
        )

        metric_3.metric(
            "Expected Peak",
            f"{peak_value:,.0f}"
        )

        st.write("")

        metric_4, coverage_1, coverage_2 = st.columns(3)

        metric_4.metric(
            "Peak Date",
            peak_date.strftime("%d %b %Y")
        )

        coverage_1.metric(
            "Forecast Period",
            "Jan–Dec 2026"
        )

        coverage_2.metric(
            "Forecast Days",
            str(forecast_days)
        )

        if (
            forecast_start
            != pd.Timestamp("2026-01-01")
            or forecast_end
            != pd.Timestamp("2026-12-31")
            or forecast_days != 365
        ):
            st.warning(
                "The forecast file does not contain all "
                "365 days of 2026."
            )

        st.html(
            """
            <div class="section-title">
                Daily Forecast
            </div>
            """
        )

        daily_figure = go.Figure()

        if (
            "yhat_upper" in forecast.columns
            and "yhat_lower" in forecast.columns
        ):
            daily_figure.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    mode="lines",
                    line={
                        "width": 0
                    },
                    hoverinfo="skip",
                    showlegend=False
                )
            )

            daily_figure.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    mode="lines",
                    line={
                        "width": 0
                    },
                    fill="tonexty",
                    fillcolor="rgba(34, 197, 94, 0.16)",
                    name="Confidence interval",
                    hovertemplate=(
                        "Date: %{x|%d %b %Y}"
                        "<br>Lower estimate: %{y:,.0f}"
                        "<extra></extra>"
                    )
                )
            )

        daily_figure.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Predicted traffic",
                line={
                    "color": "#4ade80",
                    "width": 3
                },
                hovertemplate=(
                    "Date: %{x|%d %b %Y}"
                    "<br>Predicted movements: %{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

        daily_figure = style_figure(
            daily_figure,
            "Daily Customs Traffic Forecast for 2026"
        )

        daily_figure.update_xaxes(
            tickformat="%b %Y",
            dtick="M1",
            title="Month"
        )

        daily_figure.update_yaxes(
            title="Predicted movements",
            rangemode="tozero"
        )

        daily_figure.update_layout(height=430)

        st.plotly_chart(
            daily_figure,
            use_container_width=True
        )

        monthly_source = forecast.copy()

        monthly_source["month_date"] = (
            monthly_source["ds"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        monthly_aggregation = {
            "yhat": "sum"
        }

        if "yhat_lower" in monthly_source.columns:
            monthly_aggregation[
                "yhat_lower"
            ] = "sum"

        if "yhat_upper" in monthly_source.columns:
            monthly_aggregation[
                "yhat_upper"
            ] = "sum"

        monthly_forecast = (
            monthly_source
            .groupby(
                "month_date",
                as_index=False
            )
            .agg(monthly_aggregation)
            .sort_values("month_date")
        )

        monthly_forecast["month_name"] = (
            monthly_forecast["month_date"]
            .dt.strftime("%b")
        )

        st.html(
            """
            <div class="section-title">
                Monthly Forecast Summary
            </div>
            """
        )

        monthly_figure = go.Figure()

        monthly_figure.add_trace(
            go.Bar(
                x=monthly_forecast["month_name"],
                y=monthly_forecast["yhat"],
                name="Monthly forecast",
                marker={
                    "color": "#22c55e",
                    "line": {
                        "color": "#4ade80",
                        "width": 1
                    }
                },
                text=monthly_forecast[
                    "yhat"
                ].round(0),
                texttemplate="%{text:,.0f}",
                textposition="outside",
                textfont={
                    "color": "#ffffff"
                },
                hovertemplate=(
                    "Month: %{x}"
                    "<br>Predicted movements: %{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

        monthly_figure = style_figure(
            monthly_figure,
            (
                "Predicted Monthly Traffic — 2026"
            )
        )

        monthly_figure.update_xaxes(
            title="Month",
            categoryorder="array",
            categoryarray=[
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]
        )

        monthly_figure.update_yaxes(
            title="Predicted movements",
            rangemode="tozero"
        )

        monthly_figure.update_layout(height=460)

        st.plotly_chart(
            monthly_figure,
            use_container_width=True
        )

        highest_month_row = monthly_forecast.loc[
            monthly_forecast["yhat"].idxmax()
        ]

        lowest_month_row = monthly_forecast.loc[
            monthly_forecast["yhat"].idxmin()
        ]

        summary_1, summary_2 = st.columns(2)

        with summary_1:
            st.html(
                f"""
                <div class="info-card">
                    <strong>Highest Forecast Month</strong>
                    <br><br>
                    {highest_month_row["month_name"]} 2026
                    <br>
                    Predicted movements:
                    {highest_month_row["yhat"]:,.0f}
                </div>
                """
            )

        with summary_2:
            st.html(
                f"""
                <div class="info-card">
                    <strong>Lowest Forecast Month</strong>
                    <br><br>
                    {lowest_month_row["month_name"]} 2026
                    <br>
                    Predicted movements:
                    {lowest_month_row["yhat"]:,.0f}
                </div>
                """
            )

        with st.expander(
            "View 2026 Forecast Data"
        ):
            display_columns = [
                column
                for column in [
                    "ds",
                    "yhat",
                    "yhat_lower",
                    "yhat_upper"
                ]
                if column in forecast.columns
            ]

            displayed_forecast = forecast[
                display_columns
            ].copy()

            displayed_forecast = (
                displayed_forecast.rename(
                    columns={
                        "ds": "Date",
                        "yhat": "Predicted Traffic",
                        "yhat_lower": "Lower Estimate",
                        "yhat_upper": "Upper Estimate"
                    }
                )
            )

            numeric_display_columns = [
                column
                for column in [
                    "Predicted Traffic",
                    "Lower Estimate",
                    "Upper Estimate"
                ]
                if column in displayed_forecast.columns
            ]

            displayed_forecast[
                numeric_display_columns
            ] = displayed_forecast[
                numeric_display_columns
            ].round(0)

            st.dataframe(
                displayed_forecast,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download 2026 forecast CSV",
            data=displayed_forecast.to_csv(index=False).encode("utf-8"),
            file_name="customs_forecast_2026.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "The forecast covers all 365 days of 2026. "
            "Predictions were generated using the trained "
            "Prophet traffic forecasting model."
        )
