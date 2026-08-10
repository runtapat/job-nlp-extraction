# -*- coding: utf-8 -*-
"""
ระบบสกัดข้อมูลประกาศรับสมัครงาน (Job Posting Information Extraction System)
วิชา NLP - แบบทดสอบเก็บคะแนน ครั้งที่ 1 (ข้อ 2)

เทคนิคที่ใช้:
1. Regex & Cleansing        -> ลบเบอร์โทร, ลิงก์, อีเมล (Noise / ข้อมูลอ่อนไหว)
2. Tokenization & Normalization -> ตัดคำภาษาไทยด้วย pythainlp, ลบ Stopwords
3. Topic Identification     -> จัดหมวดหมู่ประเภทงาน (IT, Marketing, Sales, ฯลฯ) ด้วย keyword scoring
4. POS & NER                -> POS Tagging (pythainlp) + Rule-based Entity Extraction
                                (ตำแหน่งงาน, บริษัท, เงินเดือน, สถานที่, ทักษะ)
"""

import re
import pandas as pd
import streamlit as st

from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบสกัดข้อมูลประกาศรับสมัครงาน",
    page_icon="🧑‍💼",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------------------------------
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Kanit', 'Inter', sans-serif;
        }

        .stApp {
            background: #0f1220;
        }
        .main .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
        }

        /* ---------- Hero ---------- */
        .hero-box {
            background: radial-gradient(circle at 15% 20%, #6d28d9 0%, #1e1b4b 55%, #0f1220 100%);
            padding: 2.4rem 2.6rem;
            border-radius: 22px;
            color: white;
            margin-bottom: 1.8rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero-box h1 {
            color: #ffffff !important;
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-box p {
            color: #c7c9f2;
            font-size: 1rem;
            margin: 0 0 0.9rem 0;
        }
        .badge-row span {
            display: inline-block;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 999px;
            padding: 5px 14px;
            margin: 4px 8px 0 0;
            font-size: 0.8rem;
            font-weight: 500;
            color: #e5e5ff;
        }

        /* ---------- General text on dark bg ---------- */
        h1, h2, h3, h4, p, label, .stMarkdown, .stCaption {
            color: #e8e8f5 !important;
        }
        .stTextArea textarea {
            background: #171a2e !important;
            color: #f1f1fa !important;
            border: 1px solid #2c2f4a !important;
            border-radius: 12px !important;
            font-family: 'Kanit', sans-serif;
        }

        /* ---------- Result cards ---------- */
        .result-card {
            background: linear-gradient(160deg, #191c33 0%, #14162a 100%);
            border: 1px solid #2c2f4a;
            border-left: 4px solid #a855f7;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
            min-height: 92px;
        }
        .result-card .label {
            font-size: 0.78rem;
            color: #9d9fc9;
            font-weight: 600;
            letter-spacing: .03em;
            text-transform: uppercase;
        }
        .result-card .value {
            font-size: 1.08rem;
            color: #ffffff !important;
            font-weight: 600;
            margin-top: 4px;
            line-height: 1.4;
        }
        .skill-chip {
            display: inline-block;
            background: rgba(168, 85, 247, 0.15);
            color: #d8b4fe !important;
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 999px;
            padding: 4px 12px;
            margin: 3px 6px 0 0;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .topic-chip {
            display: inline-block;
            background: linear-gradient(120deg, #ec4899, #a855f7);
            color: white !important;
            border-radius: 999px;
            padding: 6px 18px;
            font-size: 0.92rem;
            font-weight: 700;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #14162a;
            border-right: 1px solid #2c2f4a;
        }
        section[data-testid="stSidebar"] * {
            color: #d8d9f0 !important;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #2c2f4a;
        }
        .stTabs [data-baseweb="tab"] {
            background: #171a2e;
            border: 1px solid #2c2f4a;
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            padding: 10px 22px;
            font-weight: 600;
            color: #b7b9de !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #7c3aed, #a855f7) !important;
            color: white !important;
        }
        .stTabs [aria-selected="true"] p {
            color: white !important;
        }

        /* ---------- Buttons ---------- */
        .stButton button {
            background: linear-gradient(120deg, #7c3aed, #ec4899);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 6px 18px rgba(124, 58, 237, 0.35);
        }
        .stButton button:hover {
            filter: brightness(1.1);
        }

        /* ---------- Expander ---------- */
        .streamlit-expanderHeader, details summary {
            background: #171a2e !important;
            border-radius: 10px !important;
            color: #e8e8f5 !important;
        }
        details {
            background: #14162a;
            border: 1px solid #2c2f4a;
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }

        /* ---------- Dataframe / file uploader ---------- */
        [data-testid="stFileUploaderDropzone"] {
            background: #171a2e;
            border: 1px dashed #4c4f7a;
            border-radius: 12px;
        }
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

STOPWORDS = set(thai_stopwords())

PROVINCES = [
    "กรุงเทพมหานคร", "กรุงเทพฯ", "กรุงเทพ", "นนทบุรี", "ปทุมธานี", "สมุทรปราการ",
    "สมุทรสาคร", "นครปฐม", "ชลบุรี", "ระยอง", "เชียงใหม่", "เชียงราย", "ขอนแก่น",
    "นครราชสีมา", "อุดรธานี", "อุบลราชธานี", "สงขลา", "ภูเก็ต", "สุราษฎร์ธานี",
    "พระนครศรีอยุธยา", "ลำปาง", "นครสวรรค์", "บุรีรัมย์",
    # เขตในกรุงเทพฯ ที่พบบ่อยในประกาศงาน
    "บางนา", "บางรัก", "สาทร", "วัฒนา", "ห้วยขวาง", "ลาดพร้าว", "จตุจักร",
    "บางกะปิ", "คลองเตย", "ปทุมวัน", "ราชเทวี", "พระโขนง", "บางขุนเทียน",
    "Bangkok", "Chonburi", "Chiang Mai", "Rayong", "Nonthaburi",
]

SKILL_KEYWORDS = [
    # ภาษาโปรแกรม / IT
    "Python", "Java", "JavaScript", "SQL", "PHP", "C++", "React", "Node.js",
    "HTML", "CSS", "Excel", "Power BI", "Photoshop", "Illustrator", "Figma",
    "SEO", "Google Ads", "Facebook Ads", "TikTok",
    # ภาษา / soft skill
    "ภาษาอังกฤษ", "ภาษาจีน", "ภาษาญี่ปุ่น", "English", "Presentation",
    "การตลาด", "บัญชี", "การเงิน", "ลูกค้าสัมพันธ์", "การขาย", "เจรจาต่อรอง",
    "บริหารทีม", "วางแผนงาน", "ขับรถได้", "ทำงานภายใต้แรงกดดันได้",
]

CATEGORY_KEYWORDS = {
    "IT / Software": ["โปรแกรมเมอร์", "Developer", "Software", "IT", "SQL", "Python",
                       "Java", "โค้ด", "ระบบ", "Programmer", "Network", "System"],
    "การตลาด (Marketing)": ["การตลาด", "Marketing", "โฆษณา", "แบรนด์", "Content",
                             "Social Media", "SEO", "แคมเปญ", "Ads"],
    "งานขาย (Sales)": ["เซลล์", "Sales", "ขาย", "ลูกค้า", "ปิดการขาย", "ยอดขาย",
                        "Presale", "โควต้า"],
    "บัญชี/การเงิน (Finance & Accounting)": ["บัญชี", "การเงิน", "Accounting",
                                               "Finance", "ภาษี", "งบการเงิน",
                                               "ตรวจสอบบัญชี"],
    "ทรัพยากรบุคคล (HR)": ["HR", "บุคคล", "สรรหา", "ฝึกอบรม", "เงินเดือนพนักงาน",
                            "Recruitment", "Human Resource"],
    "กราฟิก/ออกแบบ (Design)": ["กราฟิก", "ออกแบบ", "Design", "Photoshop",
                                "Illustrator", "Figma", "UX", "UI"],
    "บริการลูกค้า (Customer Service)": ["Call Center", "บริการลูกค้า", "Customer Service",
                                          "รับเรื่องร้องเรียน", "Support"],
}


# ---------------------------------------------------------------------------
# STEP 1: Regex & Cleansing
# ---------------------------------------------------------------------------
def clean_text(text: str):
    """ลบลิงก์, อีเมล, เบอร์โทร (Noise / PII) และช่องว่างเกิน"""
    removed = {}

    urls = re.findall(r"https?://\S+|www\.\S+", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " ", text)

    phones = re.findall(r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}", text)
    text = re.sub(r"0\d{1,2}[-\s]?\d{3}[-\s]?\d{3,4}", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    removed["ลิงก์ (URLs)"] = urls
    removed["อีเมล"] = emails
    removed["เบอร์โทรศัพท์"] = phones
    return text, removed


# ---------------------------------------------------------------------------
# STEP 2: Tokenization & Normalization
# ---------------------------------------------------------------------------
def tokenize_and_normalize(text: str):
    tokens = word_tokenize(text, engine="newmm")
    tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# STEP 3: Topic Identification
# ---------------------------------------------------------------------------
def identify_topic(text: str):
    scores = {}
    text_l = text.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in kws if kw.lower() in text_l)
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "อื่นๆ (Other)", scores
    return best, scores


# ---------------------------------------------------------------------------
# STEP 4: POS & Rule-based NER
# ---------------------------------------------------------------------------
def extract_salary(text: str) -> str:
    patterns = [
        r"เงินเดือน[:\s]*([\d,]+\s*-\s*[\d,]+)\s*(บาท)?",
        r"เงินเดือน[:\s]*([\d,]+)\s*(บาท)?",
        r"salary[:\s]*([\d,]+\s*-\s*[\d,]+)",
        r"([\d,]{4,}\s*-\s*[\d,]{4,})\s*บาท",
        r"([\d,]{4,})\s*บาท",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return "ไม่พบข้อมูล"


POSITION_STOP_WORDS = r"(?:ประจำ|เงินเดือน|สาขา|จังหวัด|ที่|บริษัท|ต้องการ|รับสมัคร|โทร|ติดต่อ|Salary|Location|Requirements?|Company)"


def extract_position(text: str) -> str:
    m = re.search(
        r"ตำแหน่ง[:\s]*([A-Za-zก-๙0-9\s]{2,40}?)(?=\s+" + POSITION_STOP_WORDS + r"|[,\.]|$)",
        text,
    )
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = re.search(
        r"position[:\s]*([A-Za-zก-๙0-9\s]{2,40}?)(?=\s+" + POSITION_STOP_WORDS + r"|[,\.]|$)",
        text,
        re.IGNORECASE,
    )
    if m and m.group(1).strip():
        return m.group(1).strip()
    return "ไม่พบข้อมูล"


def extract_company(text: str) -> str:
    m = re.search(r"บริษัท\s*([^\n,\.]{2,40}?)\s*จำกัด", text)
    if m:
        return "บริษัท " + m.group(1).strip() + " จำกัด"
    m = re.search(r"company[:\s]*([^\n,\.]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "ไม่พบข้อมูล"


def extract_location(text: str) -> str:
    found = [p for p in PROVINCES if p in text]
    found = list(dict.fromkeys(found))  # unique, keep order
    return ", ".join(found) if found else "ไม่พบข้อมูล"


def extract_skills(text: str) -> str:
    """ใช้ word boundary (\\b) สำหรับคำภาษาอังกฤษ เพื่อกันปัญหาคำซ้อนกัน เช่น
    'Java' ที่เป็นส่วนหนึ่งของ 'JavaScript' ส่วนคำภาษาไทยใช้ substring match
    เนื่องจากภาษาไทยไม่มีช่องว่างคั่นคำ ทำให้ \\b ใช้ไม่ได้ผลตามปกติ"""
    found = []
    for s in SKILL_KEYWORDS:
        if s.isascii():
            if re.search(r"\b" + re.escape(s) + r"\b", text, re.IGNORECASE):
                found.append(s)
        else:
            if s in text:
                found.append(s)
    return ", ".join(found) if found else "ไม่พบข้อมูล"


# ---------------------------------------------------------------------------
# Pipeline รวมทุกขั้นตอน
# ---------------------------------------------------------------------------
def process_text(raw_text: str):
    cleaned, removed = clean_text(raw_text)
    tokens = tokenize_and_normalize(cleaned)
    topic, scores = identify_topic(cleaned)

    try:
        tags = pos_tag(tokens, engine="perceptron", corpus="orchid") if tokens else []
    except Exception:
        tags = []

    result = {
        "ตำแหน่งงาน": extract_position(cleaned),
        "บริษัท": extract_company(cleaned),
        "เงินเดือน": extract_salary(cleaned),
        "สถานที่": extract_location(cleaned),
        "ทักษะที่ต้องการ": extract_skills(cleaned),
        "หมวดหมู่งาน": topic,
    }
    return result, cleaned, tokens, tags, removed, scores


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-box">
        <h1>🧑‍💼 ระบบสกัดข้อมูลประกาศรับสมัครงาน</h1>
        <p>Job Posting Information Extraction — วิเคราะห์ข้อความประกาศรับสมัครงานภาษาไทย/อังกฤษด้วย NLP</p>
        <div class="badge-row">
            <span>🧹 Regex & Cleansing</span>
            <span>✂️ Tokenization</span>
            <span>🗂️ Topic Identification</span>
            <span>🏷️ POS & NER</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📌 เกี่ยวกับระบบ")
    st.write(
        "อัปโหลดหรือวางข้อความประกาศรับสมัครงาน ระบบจะทำความสะอาดข้อความ "
        "ตัดคำ จัดหมวดหมู่ และสกัดข้อมูลสำคัญ เช่น ตำแหน่งงาน บริษัท เงินเดือน "
        "สถานที่ทำงาน และทักษะที่ต้องการ โดยอัตโนมัติ"
    )
    st.markdown("### ⚙️ เทคนิคที่ใช้")
    st.markdown(
        "- 🧹 **Regex & Cleansing**\n"
        "- ✂️ **Tokenization & Normalization** (pythainlp)\n"
        "- 🗂️ **Topic Identification** (keyword scoring)\n"
        "- 🏷️ **POS Tagging & Rule-based NER**"
    )
    st.markdown("---")
    st.caption("จัดทำเพื่อการศึกษา รายวิชา NLP")

tab1, tab2 = st.tabs(["📝  ทดลองข้อความเดียว", "📂  ประมวลผลไฟล์ (หลายประกาศ)"])

# ---------------- TAB 1: single text ----------------
with tab1:
    sample_text = (
        "บริษัท ทีเอชสมาร์ทเทค จำกัด เปิดรับสมัครตำแหน่ง Frontend Developer "
        "ประจำสาขาบางนา กรุงเทพฯ เงินเดือน 25,000-35,000 บาท "
        "ต้องการผู้มีทักษะ JavaScript, React, HTML, CSS และภาษาอังกฤษในการสื่อสาร "
        "สนใจติดต่อ 081-234-5678 หรืออีเมล hr@thsmarttech.com "
        "ดูรายละเอียดเพิ่มเติมที่ https://thsmarttech.com/careers"
    )

    text_input = st.text_area(
        "วางข้อความประกาศรับสมัครงาน (ไทย/อังกฤษ)",
        value=sample_text,
        height=180,
    )

    if st.button("🔍 ประมวลผล", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("กรุณาใส่ข้อความก่อนประมวลผล")
        else:
            result, cleaned, tokens, tags, removed, scores = process_text(text_input)

            st.markdown("### ✅ ผลลัพธ์การสกัดข้อมูล")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="result-card"><div class="label">💼 ตำแหน่งงาน</div>
                    <div class="value">{result['ตำแหน่งงาน']}</div></div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""<div class="result-card"><div class="label">🏢 บริษัท</div>
                    <div class="value">{result['บริษัท']}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="result-card"><div class="label">💰 เงินเดือน</div>
                    <div class="value">{result['เงินเดือน']}</div></div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""<div class="result-card"><div class="label">📍 สถานที่</div>
                    <div class="value">{result['สถานที่']}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                skills_html = "".join(
                    f'<span class="skill-chip">{s}</span>'
                    for s in result["ทักษะที่ต้องการ"].split(", ")
                    if s != "ไม่พบข้อมูล"
                ) or "ไม่พบข้อมูล"
                st.markdown(
                    f"""<div class="result-card"><div class="label">🧠 ทักษะที่ต้องการ</div>
                    <div class="value">{skills_html}</div></div>""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""<div class="result-card"><div class="label">🗂️ หมวดหมู่งาน</div>
                    <div class="value"><span class="topic-chip">{result['หมวดหมู่งาน']}</span></div></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("")
            with st.expander("🧹 ขั้นตอนที่ 1: Regex & Cleansing (ข้อมูลที่ถูกลบออก)"):
                for k, v in removed.items():
                    st.write(f"**{k}:** {v if v else '- ไม่พบ -'}")
                st.write("**ข้อความหลังทำความสะอาด:**")
                st.info(cleaned)

            with st.expander("✂️ ขั้นตอนที่ 2: Tokenization & Normalization"):
                st.write(f"จำนวนคำหลังตัดคำและลบ Stopwords: {len(tokens)}")
                st.write(tokens)

            with st.expander("🗂️ ขั้นตอนที่ 3: Topic Identification (คะแนนแต่ละหมวดหมู่)"):
                st.bar_chart(pd.Series(scores))

            with st.expander("🏷️ ขั้นตอนที่ 4: POS Tagging"):
                if tags:
                    st.dataframe(pd.DataFrame(tags, columns=["คำ", "POS Tag"]), use_container_width=True)
                else:
                    st.write("ไม่มีข้อมูลคำสำหรับติด POS Tag")

# ---------------- TAB 2: batch file ----------------
with tab2:
    st.write(
        "อัปโหลดไฟล์ CSV ที่มีคอลัมน์ชื่อ `text` (แต่ละแถวคือประกาศงาน 1 รายการ) "
        "หรือใช้ไฟล์ตัวอย่าง `sample_data.csv` ที่แนบมากับโปรเจกต์นี้"
    )
    uploaded = st.file_uploader("อัปโหลดไฟล์ CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        if "text" not in df_in.columns:
            st.error("ไฟล์ CSV ต้องมีคอลัมน์ชื่อ 'text'")
        else:
            if st.button("🔍 ประมวลผลทั้งไฟล์", type="primary"):
                rows = []
                progress = st.progress(0)
                for i, t in enumerate(df_in["text"].astype(str)):
                    result, *_ = process_text(t)
                    rows.append(result)
                    progress.progress((i + 1) / len(df_in))
                out_df = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
                st.subheader("✅ ผลลัพธ์ทั้งหมด")
                st.dataframe(out_df, use_container_width=True)
                st.download_button(
                    "⬇️ ดาวน์โหลดผลลัพธ์เป็น CSV",
                    out_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="job_extraction_results.csv",
                    mime="text/csv",
                )

st.divider()
st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.85rem;'>"
    "จัดทำเพื่อการศึกษา รายวิชา NLP — เทคนิค Regex · Tokenization · Topic Identification · POS & NER"
    "</p>",
    unsafe_allow_html=True,
)
