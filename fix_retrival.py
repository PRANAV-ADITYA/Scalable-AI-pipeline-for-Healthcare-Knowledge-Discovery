# Quick test to identify the exact pgvector query issue
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import numpy as np

PG_URL = "postgresql://medai:medai123@localhost:5432/medical_insights"
engine = create_engine(PG_URL)
model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query     = "cancer radiation treatment"
query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

# Test different vector format approaches
print("Testing vector format...")

# Format 1: direct string
vec_str1 = "[" + ",".join(str(v) for v in query_vec) + "]"
print(f"Format 1 preview: {vec_str1[:50]}...")

try:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.summary, s.rouge_score, s.topic_label,
                   (vs.embedding <#> CAST(:qvec AS vector)) * -1 AS similarity
            FROM summary_vectors vs
            JOIN validated_summaries s ON vs.abstract_id = s.abstract_id
            ORDER BY vs.embedding <#> CAST(:qvec AS vector)
            LIMIT 3
        """), {"qvec": vec_str1}).fetchall()
        print(f"Format 1 SUCCESS — got {len(rows)} rows")
        for r in rows:
            print(f"  [{r.topic_label}] {r.summary[:80]}...")
except Exception as e:
    print(f"Format 1 FAILED: {e}")

# Format 2: numpy array to string
try:
    vec_arr = np.array(query_vec, dtype=np.float32)
    vec_str2 = str(vec_arr.tolist())
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.summary, s.rouge_score
            FROM summary_vectors vs
            JOIN validated_summaries s ON vs.abstract_id = s.abstract_id
            ORDER BY vs.embedding <#> :qvec::vector
            LIMIT 3
        """), {"qvec": vec_str2}).fetchall()
        print(f"Format 2 SUCCESS — got {len(rows)} rows")
except Exception as e:
    print(f"Format 2 FAILED: {e}")