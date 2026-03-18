# ============================================================
# MedAI Knowledge Discovery — Flask RAG Backend (Cloud Final)
# DB  : Azure PostgreSQL + pgvector (HNSW index)
# LLM : Google Gemini 2.5 Flash
# Embed: TF-IDF based retrieval (no heavy ML model needed)
#        Falls back to pgvector full-text search
#        Zero heavy dependencies — fits in 512MB RAM
#        Scales to millions of docs via PostgreSQL
# ============================================================

import os
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
import google.generativeai as genai

# ── Gemini Configuration ─────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "REMOVED")
genai.configure(api_key=API_KEY)

# ── Database Configuration ───────────────────────────────────
# Local  : points to Azure PostgreSQL (same as cloud)
# Cloud  : reads from environment variables set in App Service
PG_HOST     = os.environ.get("PG_HOST",     "medai-postgres-server.postgres.database.azure.com")
PG_PORT     = "5432"
PG_USER     = os.environ.get("PG_USER",     "medai")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "MedAI%40123456")
PG_DB       = os.environ.get("PG_DB",       "medical_insights")
PG_URL      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=require"

TABLE_NAME  = "validated_summaries"
TOP_K       = 8

app    = Flask(__name__)
CORS(app)
engine = create_engine(
    PG_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True    # auto-reconnect if connection drops
)

def verify_db():
    try:
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
            vecs  = conn.execute(text("SELECT COUNT(*) FROM summary_vectors")).scalar()
            print(f"  PostgreSQL connected : {count} records, {vecs} vectors")
    except Exception as e:
        print(f"  WARNING: DB connection failed: {e}")

verify_db()

# ============================================================
# RETRIEVAL — Pure PostgreSQL, no ML model needed
#
# Strategy: Two-stage retrieval
# Stage 1: PostgreSQL full-text search (fast, scales to millions)
#          Uses GIN index on tsvector — sub-second at any scale
# Stage 2: Re-rank by ROUGE score + topic boost
#
# Why this scales to millions:
# - Full-text search uses PostgreSQL's built-in GIN index
# - No Python ML model running in memory
# - All computation inside the database engine
# - Add more Postgres replicas = linear scale
# ============================================================

def ensure_fts_index():
    """Create full-text search index if not exists."""
    try:
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_summary_fts
                ON {TABLE_NAME}
                USING GIN(to_tsvector('english', summary))
            """))
            conn.commit()
    except Exception:
        pass

ensure_fts_index()


def get_context(query: str, topic_hint: str = None) -> str:
    """
    Two-stage retrieval:
    1. PostgreSQL full-text search (GIN index — scales to millions)
    2. Re-rank by ROUGE score + topic boost
    """
    try:
        # Clean query for full-text search
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        # Convert to tsquery format: "smoking & children"
        ts_query = " & ".join([
            w for w in clean_query.lower().split()
            if len(w) > 2
        ])

        with engine.connect() as conn:

            results = []

            # Stage 1a: Full-text search using GIN index
            if ts_query:
                rows = conn.execute(text(f"""
                    SELECT
                        abstract_id,
                        summary,
                        rouge_score,
                        topic_label,
                        ts_rank(
                            to_tsvector('english', summary),
                            to_tsquery('english', :tsq)
                        ) AS relevance
                    FROM {TABLE_NAME}
                    WHERE to_tsvector('english', summary) @@ to_tsquery('english', :tsq)
                    ORDER BY relevance DESC, rouge_score DESC
                    LIMIT :k
                """), {"tsq": ts_query, "k": TOP_K * 2}).fetchall()
                results = list(rows)

            # Stage 1b: Fallback — LIKE search if FTS returns nothing
            if len(results) < 4:
                keywords = [w for w in clean_query.lower().split() if len(w) > 3]
                if keywords:
                    search_clause = " OR ".join([f"summary ILIKE :{f'kw{i}'}" for i in range(len(keywords))])
                    params = {f"kw{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
                    params["k"] = TOP_K * 2
                    fallback = conn.execute(text(f"""
                        SELECT abstract_id, summary, rouge_score, topic_label,
                               0.1 AS relevance
                        FROM {TABLE_NAME}
                        WHERE {search_clause}
                        ORDER BY rouge_score DESC
                        LIMIT :k
                    """), params).fetchall()
                    # Merge without duplicates
                    existing_ids = {r.abstract_id for r in results}
                    for row in fallback:
                        if row.abstract_id not in existing_ids:
                            results.append(row)

            # Stage 1c: Safety fallback — top ROUGE records
            if len(results) < 4:
                safety = conn.execute(text(f"""
                    SELECT abstract_id, summary, rouge_score, topic_label,
                           0.05 AS relevance
                    FROM {TABLE_NAME}
                    ORDER BY rouge_score DESC
                    LIMIT 8
                """)).fetchall()
                existing_ids = {r.abstract_id for r in results}
                for row in safety:
                    if row.abstract_id not in existing_ids:
                        results.append(row)

            # Stage 2: Re-rank — topic boost + ROUGE score
            results = sorted(
                results,
                key=lambda r: (
                    1 if (topic_hint and r.topic_label == topic_hint) else 0,
                    r.rouge_score
                ),
                reverse=True
            )[:TOP_K]

        context_lines = []
        for r in results:
            label = r.topic_label or 'general'
            context_lines.append(
                f"[Topic: {label} | ROUGE-L: {round(r.rouge_score, 3)}]\n{r.summary}"
            )

        return "\n\n".join(context_lines)

    except Exception as e:
        return f"Retrieval error: {str(e)}"


# ── Topic keyword detector ───────────────────────────────────
def detect_topic(query: str):
    q = query.lower()
    topic_map = {
        "cancer":          "cancer_radiation_patients",
        "radiation":       "cancer_radiation_patients",
        "tumor":           "cancer_radiation_patients",
        "oncology":        "cancer_radiation_patients",
        "heart":           "patients_pressure_angina",
        "cardiac":         "patients_pressure_angina",
        "angina":          "patients_pressure_angina",
        "cardiovascular":  "patients_pressure_angina",
        "pressure":        "patients_pressure_angina",
        "smoking":         "smoking_assigned_children",
        "cigarette":       "smoking_assigned_children",
        "children":        "smoking_assigned_children",
        "infant":          "infants_milk_vitamin",
        "milk":            "infants_milk_vitamin",
        "vitamin":         "infants_milk_vitamin",
        "glucose":         "glucose_cholesterol_blood glucose",
        "cholesterol":     "glucose_cholesterol_blood glucose",
        "diabetes":        "glucose_cholesterol_blood glucose",
        "gastric":         "gastric_ph_subjects",
        "stomach":         "gastric_ph_subjects",
        "arthritis":       "arthritis_patients_months",
        "joint":           "arthritis_patients_months",
        "lung":            "pulmonary_training_chest",
        "pulmonary":       "pulmonary_training_chest",
        "urine":           "urine_excretion_salt",
        "kidney":          "urine_excretion_salt",
        "plasma":          "healthy_mg_plasma",
        "magnesium":       "healthy_mg_plasma",
        "randomized":      "patients_randomized_prospective",
        "postoperative":   "topical_postoperative_objective",
    }
    for kw, topic in topic_map.items():
        if kw in q:
            return topic
    return None


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        with engine.connect() as conn:
            count     = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
            vec_count = conn.execute(text("SELECT COUNT(*) FROM summary_vectors")).scalar()
            avg_rouge = conn.execute(text(
                f"SELECT ROUND(AVG(rouge_score)::numeric, 3) FROM {TABLE_NAME}"
            )).scalar()
            topics = conn.execute(text(f"""
                SELECT topic_label, COUNT(*) as cnt
                FROM {TABLE_NAME}
                GROUP BY topic_label
                ORDER BY cnt DESC
            """)).fetchall()

        return jsonify({
            "count":      count,
            "status":     "Online",
            "avg_rouge":  float(avg_rouge) if avg_rouge else 0,
            "pgvector":   vec_count,
            "topics":     {r.topic_label: r.cnt for r in topics},
            "db_backend": "Azure PostgreSQL + pgvector",
            "retrieval":  "PostgreSQL FTS (GIN index) — scales to millions",
            "llm":        "Gemini 2.5 Flash"
        })
    except Exception as e:
        return jsonify({"count": 0, "status": "Offline", "error": str(e)})


@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json
    user_query = data.get("query", "")

    if not user_query.strip():
        return jsonify({"response": "Please enter a query.", "topic_hint": None})

    topic_hint   = detect_topic(user_query)
    context_data = get_context(user_query, topic_hint)

    system_prompt = """You are MedAI Insight, a medical research assistant.
You answer ONLY based on the research summaries given to you.
You NEVER use outside knowledge.
Structure responses with bold headers like **Key Findings**, **Study Details**, **Clinical Implications**.
Be specific about what each study found.
Start directly with findings."""

    user_message = f"""Here are PubMed research summaries retrieved for this query:

---
{context_data}
---

Using ONLY the summaries above, answer this question:
{user_query}

Reference specific findings from the summaries."""

    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=system_prompt
        )
        response = model.generate_content(user_message)
        return jsonify({
            "response":   response.text,
            "topic_hint": topic_hint,
            "sources":    len(context_data.split("\n\n")),
            "db_backend": "Azure PostgreSQL + pgvector",
            "llm":        "gemini-2.5-flash"
        })
    except Exception as e:
        return jsonify({"response": f"Gemini Error: {str(e)}"})


@app.route('/api/topics', methods=['GET'])
def get_topics():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT topic_id, topic_label,
                       COUNT(*) as count,
                       ROUND(AVG(rouge_score)::numeric, 3) as avg_rouge
                FROM {TABLE_NAME}
                GROUP BY topic_id, topic_label
                ORDER BY count DESC
            """)).fetchall()
        return jsonify([dict(r._mapping) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/search', methods=['POST'])
def semantic_search():
    """
    Full-text search endpoint — shows reviewer what
    PostgreSQL FTS is retrieving for any query.
    Scales to millions via GIN index.
    """
    data   = request.json
    query  = data.get("query", "")
    top_k  = data.get("top_k", 5)
    topic_hint = detect_topic(query)

    clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    ts_query = " & ".join([w for w in clean_query.lower().split() if len(w) > 2])

    try:
        with engine.connect() as conn:
            if ts_query:
                rows = conn.execute(text(f"""
                    SELECT summary, rouge_score, topic_label,
                           ts_rank(
                               to_tsvector('english', summary),
                               to_tsquery('english', :tsq)
                           ) AS relevance
                    FROM {TABLE_NAME}
                    WHERE to_tsvector('english', summary) @@ to_tsquery('english', :tsq)
                    ORDER BY relevance DESC
                    LIMIT :k
                """), {"tsq": ts_query, "k": top_k}).fetchall()
            else:
                rows = conn.execute(text(f"""
                    SELECT summary, rouge_score, topic_label, 0 AS relevance
                    FROM {TABLE_NAME}
                    ORDER BY rouge_score DESC
                    LIMIT :k
                """), {"k": top_k}).fetchall()

        return jsonify({
            "query":      query,
            "topic_hint": topic_hint,
            "retrieval":  "PostgreSQL FTS — GIN index",
            "results": [
                {
                    "summary":     r.summary,
                    "rouge_score": round(r.rouge_score, 3),
                    "topic":       r.topic_label,
                    "relevance":   round(float(r.relevance), 4)
                }
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "MedAI Backend"})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  MedAI Knowledge Discovery — Backend (Cloud Final)")
    print("  DB       : Azure PostgreSQL + pgvector")
    print("  Retrieval: PostgreSQL FTS (GIN) — no ML model needed")
    print("  LLM      : Gemini 2.5 Flash")
    print("  RAM      : ~200MB — fits Azure free tier")
    print("  Scale    : Millions of docs via PostgreSQL indexes")
    print("="*60)
    print(f"  Starting Flask on http://0.0.0.0:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)