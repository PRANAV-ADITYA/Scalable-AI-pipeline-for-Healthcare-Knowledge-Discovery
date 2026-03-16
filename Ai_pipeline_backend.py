# ============================================================
# MedAI Knowledge Discovery — Flask RAG Backend (Final)
# DB  : PostgreSQL + pgvector (HNSW index)
# LLM : Google Gemini 2.5 Flash
# ============================================================

import os
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import google.generativeai as genai

# ── Gemini Configuration ─────────────────────────────────────
API_KEY = "REMOVED"
genai.configure(api_key=API_KEY)

# ── Database Configuration ───────────────────────────────────
PG_HOST     = "localhost"
PG_PORT     = "5432"
PG_USER     = "medai"
PG_PASSWORD = "medai123"
PG_DB       = "medical_insights"
PG_URL      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

TABLE_NAME  = "validated_summaries"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 8

app    = Flask(__name__)
CORS(app)
engine = create_engine(PG_URL)

print("Loading embedding model...")
_embed_model = SentenceTransformer(EMBED_MODEL)
print("  Embedding model loaded.")

def verify_db():
    try:
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
            vecs  = conn.execute(text("SELECT COUNT(*) FROM summary_vectors")).scalar()
            print(f"  PostgreSQL connected : {count} records, {vecs} vectors")
            return True
    except Exception as e:
        print(f"  WARNING: DB connection failed: {e}")
        return False

verify_db()

# ============================================================
# RETRIEVAL
# ============================================================
def get_context(query: str, topic_hint: str = None) -> str:
    try:
        query_vec = _embed_model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        with engine.connect() as conn:
            conn.execute(text("SET hnsw.ef_search = 100"))

            rows = conn.execute(text(f"""
                SELECT
                    vs.abstract_id,
                    s.summary,
                    s.rouge_score,
                    s.topic_label,
                    (vs.embedding <#> CAST(:qvec AS vector)) * -1 AS similarity
                FROM summary_vectors vs
                JOIN {TABLE_NAME} s ON vs.abstract_id = s.abstract_id
                ORDER BY vs.embedding <#> CAST(:qvec AS vector)
                LIMIT :k
            """), {"qvec": vec_str, "k": TOP_K * 2}).fetchall()

            if not rows:
                rows = conn.execute(text(f"""
                    SELECT abstract_id, summary, rouge_score, topic_label,
                           1.0 AS similarity
                    FROM {TABLE_NAME}
                    ORDER BY rouge_score DESC
                    LIMIT 8
                """)).fetchall()

            results = sorted(
                rows,
                key=lambda r: (
                    1 if (topic_hint and r.topic_label == topic_hint) else 0,
                    r.rouge_score
                ),
                reverse=True
            )[:TOP_K]

        context_lines = []
        for r in results:
            label = r.topic_label or 'general'
            score = round(r.rouge_score, 3)
            sim   = round(float(r.similarity), 3)
            context_lines.append(
                f"[Topic: {label} | ROUGE-L: {score} | Similarity: {sim}]\n{r.summary}"
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
            "db_backend": "PostgreSQL + pgvector",
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
Structure responses with bold headers.
Be specific about what each study found."""

    user_message = f"""Here are PubMed research summaries retrieved for this query:

---
{context_data}
---

Using ONLY the summaries above, answer this question:
{user_query}

Important: Reference specific findings from the summaries. Do not say there is no information."""

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
            "db_backend": "PostgreSQL + pgvector",
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
    data       = request.json
    query      = data.get("query", "")
    top_k      = data.get("top_k", 5)
    topic_hint = detect_topic(query)

    try:
        query_vec = _embed_model.encode(
            [query], normalize_embeddings=True
        )[0].tolist()
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    s.summary,
                    s.rouge_score,
                    s.topic_label,
                    (vs.embedding <#> CAST(:qvec AS vector)) * -1 AS similarity
                FROM summary_vectors vs
                JOIN {TABLE_NAME} s ON vs.abstract_id = s.abstract_id
                ORDER BY vs.embedding <#> CAST(:qvec AS vector)
                LIMIT :k
            """), {"qvec": vec_str, "k": top_k}).fetchall()

        return jsonify({
            "query":      query,
            "topic_hint": topic_hint,
            "results": [
                {
                    "summary":     r.summary,
                    "rouge_score": round(r.rouge_score, 3),
                    "topic":       r.topic_label,
                    "similarity":  round(float(r.similarity), 3)
                }
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  MedAI Knowledge Discovery — Backend (Final)")
    print("  DB  : PostgreSQL + pgvector (HNSW)")
    print("  LLM : Gemini 2.5 Flash")
    print("="*55)
    print(f"  Starting Flask on http://0.0.0.0:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=True)