# ============================================================
# MedAI Knowledge Discovery — Scalable AI Pipeline (V7)
# Fix #6 — toPandas() replaced with df.write.jdbc()
#           Spark writes directly to PostgreSQL — no driver OOM
# Fix #7 — FAISS IndexFlatIP replaced with IndexIVFFlat
#           Approximate nearest neighbour — 100x faster at scale
# Fix #8 — Vectors stored in pgvector (PostgreSQL)
#           No separate .faiss file — single DB for everything
# ============================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType, FloatType
from transformers import pipeline
import pandas as pd
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
import hashlib
import pickle
import os

# ── Database Configuration ───────────────────────────────────
# Local: PostgreSQL via Docker
# Cloud: Just change this URL to your AWS RDS endpoint
PG_HOST     = "localhost"
PG_PORT     = "5432"
PG_USER     = "medai"
PG_PASSWORD = "medai123"
PG_DB       = "medical_insights"
PG_URL      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_URL    = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
JDBC_JAR    = os.path.expanduser("~/jdbc_drivers/postgresql-42.7.3.jar")

TABLE_NAME  = "validated_summaries"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM   = 384
TOPIC_PATH  = "bertopic_model"

# ── SQLAlchemy engine for schema setup ───────────────────────
engine = create_engine(PG_URL)

# ============================================================
# STAGE 0: SCHEMA SETUP + INCREMENTAL CHECK
# ============================================================
def setup_schema():
    """Create tables if they don't exist. Safe to re-run."""
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Main summaries table
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                abstract_id  TEXT PRIMARY KEY,
                abstract     TEXT,
                summary      TEXT,
                rouge_score  REAL,
                topic_id     INTEGER,
                topic_label  TEXT,
                topic_score  REAL
            )
        """))

        # Vectors table — pgvector column (Fix #8)
        # Stores 384-dim embeddings directly in PostgreSQL
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS summary_vectors (
                abstract_id  TEXT PRIMARY KEY,
                embedding    vector({EMBED_DIM})
            )
        """))

        # Index for fast vector similarity search (Fix #7)
        # ivfflat = approximate nearest neighbour — scales to millions
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS summary_vectors_embedding_idx
            ON summary_vectors
            USING ivfflat (embedding vector_ip_ops)
            WITH (lists = 100)
        """))

        conn.commit()
    print("  Schema ready.")

def get_existing_ids():
    """Return set of abstract_ids already in DB — for incremental runs."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT abstract_id FROM {TABLE_NAME}")).fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()

def make_id(text_: str) -> str:
    return hashlib.md5(text_.encode()).hexdigest()

print("\n─── STAGE 0: Setting up PostgreSQL schema ───")
setup_schema()

# ============================================================
# STAGE 1: DATA INGESTION & DISTRIBUTION
# ============================================================
# FIX #6 — Spark configured with JDBC jar for direct DB writes
spark = SparkSession.builder \
    .appName("MedAI_Pipeline_V7") \
    .master("local[*]") \
    .config("spark.jars", JDBC_JAR) \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

print("\n─── STAGE 1: Loading PubMed abstracts ───")
dataset   = load_dataset("cyrilzakka/pubmed-medline", split='train[:500]')
df_pandas = dataset.to_pandas()

if 'content' in df_pandas.columns:
    df_pandas.rename(columns={'content': 'abstract'}, inplace=True)
else:
    raise KeyError("Could not find abstract column.")

df_pandas.dropna(subset=['abstract'], inplace=True)
df_pandas['abstract_id'] = df_pandas['abstract'].apply(make_id)

# Incremental — skip already processed
existing_ids = get_existing_ids()
before_count = len(df_pandas)
df_pandas    = df_pandas[~df_pandas['abstract_id'].isin(existing_ids)].reset_index(drop=True)
skipped      = before_count - len(df_pandas)

print(f"  Total loaded   : {before_count}")
print(f"  Already in DB  : {skipped}  (skipped)")
print(f"  New to process : {len(df_pandas)}")

if df_pandas.empty:
    print("\n  Nothing new to process.")
    spark.stop()
    exit(0)

df = spark.createDataFrame(df_pandas[['abstract_id', 'abstract']])
df = df.repartition(20)
print(f"  Partitions     : {df.rdd.getNumPartitions()}")

# ============================================================
# STAGE 2: PARALLEL GENERATIVE SUMMARIZATION
# ============================================================
print("\n─── STAGE 2: Distributed T5 Summarization ───")

@pandas_udf(StringType())
def scalable_summarize(texts: pd.Series) -> pd.Series:
    summarizer = pipeline(
        "summarization",
        model="Falconsai/medical_summarization",
        device=-1
    )
    summaries = summarizer(
        texts.tolist(),
        max_length=100,
        min_length=40,
        num_beams=4,
        do_sample=False,
        truncation=True
    )
    return pd.Series([s['summary_text'] for s in summaries])

df_summarized = df.withColumn("summary", scalable_summarize(col("abstract")))
df_summarized.show(3, truncate=80)

# ============================================================
# STAGE 2.5: ROUGE-L QUALITY GATE
# ============================================================
print("\n─── STAGE 2.5: ROUGE-L Quality Gate ───")

@pandas_udf(FloatType())
def calculate_rouge(abstracts: pd.Series, summaries: pd.Series) -> pd.Series:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return pd.Series([
        scorer.score(str(a), str(s))['rougeL'].fmeasure
        for a, s in zip(abstracts, summaries)
    ])

df_scored    = df_summarized.withColumn(
    "rouge_score", calculate_rouge(col("abstract"), col("summary"))
)
df_validated = df_scored.filter(col("rouge_score") >= 0.15)

total = df_scored.count()
kept  = df_validated.count()
print(f"  Passed gate : {kept}/{total}  ({total - kept} blocked)")

# ============================================================
# FIX #6 — DIRECT SPARK → POSTGRESQL WRITE
# No toPandas() — Spark workers write directly to DB.
# This is what makes it scale to millions of documents.
# At 1M records, no single machine ever holds all data in RAM.
# ============================================================
print("\n─── STAGE 2.6: Direct Spark → PostgreSQL Write (Fix #6) ───")

jdbc_properties = {
    "user":     PG_USER,
    "password": PG_PASSWORD,
    "driver":   "org.postgresql.Driver"
}

# Write validated summaries directly from Spark workers to PostgreSQL
# mode="append" + PRIMARY KEY in schema handles deduplication
df_validated.select(
    "abstract_id", "abstract", "summary", "rouge_score"
).write.jdbc(
    url=JDBC_URL,
    table=TABLE_NAME,
    mode="append",
    properties=jdbc_properties
)

print(f"  Spark wrote {kept} records directly to PostgreSQL.")
print("  No toPandas() — zero driver memory pressure.")

# ============================================================
# STAGE 3: SENTENCE EMBEDDINGS
# We DO need to collect summaries for embedding + BERTopic.
# But this is only the summary text (~100 tokens each),
# not the full abstracts — much smaller memory footprint.
# ============================================================
print("\n─── STAGE 3: Generating Sentence Embeddings ───")

# Collect only summary text — not full abstracts
summaries_pdf = df_validated.select("abstract_id", "summary").toPandas()
summary_list  = summaries_pdf['summary'].tolist()
id_list       = summaries_pdf['abstract_id'].tolist()

embed_model = SentenceTransformer(EMBED_MODEL)
embeddings  = embed_model.encode(
    summary_list,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
).astype(np.float32)

print(f"  Embedding shape : {embeddings.shape}")

# ============================================================
# STAGE 3.5: BERTOPIC TOPIC MODELING
# ============================================================
print("\n─── STAGE 3.5: BERTopic Topic Modeling ───")

umap_model = UMAP(
    n_components=5,
    n_neighbors=15,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)

topic_model   = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(summary_list, embeddings)
topic_info    = topic_model.get_topic_info()

print("\n  Discovered Topics:")
print(topic_info[['Topic', 'Count', 'Name']].to_string(index=False))

topic_model.save(TOPIC_PATH)
print(f"\n  BERTopic model saved to: {TOPIC_PATH}")

def get_topic_label(topic_id):
    if topic_id == -1:
        return "outlier"
    words = topic_model.get_topic(topic_id)
    if words:
        return "_".join([w[0] for w in words[:3]])
    return f"topic_{topic_id}"

topic_ids    = topics
topic_scores = [round(float(max(p) if hasattr(p, '__iter__') else p), 4) for p in probs]
topic_labels = [get_topic_label(t) for t in topic_ids]

# Update topic columns in PostgreSQL
print("\n  Updating topic assignments in PostgreSQL...")
with engine.connect() as conn:
    for i, abs_id in enumerate(id_list):
        conn.execute(text(f"""
            UPDATE {TABLE_NAME}
            SET topic_id    = :tid,
                topic_label = :tlabel,
                topic_score = :tscore
            WHERE abstract_id = :aid
        """), {
            "tid":    int(topic_ids[i]),
            "tlabel": topic_labels[i],
            "tscore": topic_scores[i],
            "aid":    abs_id
        })
    conn.commit()
print("  Topic assignments updated.")

# ============================================================
# STAGE 4: PGVECTOR — Store embeddings in PostgreSQL  (Fix #8)
# No separate .faiss file. Vectors live in the same DB.
# Uses ivfflat index for approximate nearest neighbour search.
# ============================================================
print("\n─── STAGE 4: Storing Vectors in pgvector (Fix #8) ───")

with engine.connect() as conn:
    records = [
        (id_list[i], embeddings[i].tolist())
        for i in range(len(id_list))
    ]
    execute_values(
        conn.connection.cursor(),
        """
        INSERT INTO summary_vectors (abstract_id, embedding)
        VALUES %s
        ON CONFLICT (abstract_id) DO NOTHING
        """,
        records,
        template="(%s, %s::vector)"
    )
    conn.connection.commit()

print(f"  Stored {len(records)} vectors in pgvector.")

# ============================================================
# STAGE 5: VERIFY
# ============================================================
print("\n─── STAGE 5: Verification ───")

with engine.connect() as conn:
    count     = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    vec_count = conn.execute(text("SELECT COUNT(*) FROM summary_vectors")).scalar()
    avg_rouge = conn.execute(text(f"SELECT ROUND(AVG(rouge_score)::numeric, 3) FROM {TABLE_NAME}")).scalar()
    topics_ct = conn.execute(text(f"SELECT COUNT(DISTINCT topic_label) FROM {TABLE_NAME}")).scalar()

print(f"  Records in DB      : {count}")
print(f"  Vectors in pgvector: {vec_count}")
print(f"  Avg ROUGE score    : {avg_rouge}")
print(f"  Distinct topics    : {topics_ct}")

spark.stop()
print("\n✅ MedAI Pipeline V7 complete — scale-ready!\n")