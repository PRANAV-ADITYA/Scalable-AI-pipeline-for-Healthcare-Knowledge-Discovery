# The COMPLETE End-to-End Scalable AI Pipeline (V4: Final Integrated Version with Validation)
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from transformers import pipeline
import pandas as pd
import os 
import nltk
from datasets import load_dataset 
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sqlite3
# NEW IMPORT REQUIRED FOR VALIDATION
from rouge_score import rouge_scorer 

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK resources: stopwords and punkt...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK downloads complete.")

# STAGE 1: DATA INGESTION & SCALABILITY
spark = SparkSession.builder \
    .appName("FullScalableAIPipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.conf.set("spark.sql.shuffle.partitions", "20") 

# Using 500 instead of 5000 so it runs fast during your live review
print("\n--- Loading real biomedical abstracts from PubMed ---")
dataset = load_dataset("cyrilzakka/pubmed-medline", split='train[:500]')
df_pandas = dataset.to_pandas()

ABSTRACT_COLUMN_NAME = 'content' 

if ABSTRACT_COLUMN_NAME in df_pandas.columns:
    df_pandas.rename(columns={ABSTRACT_COLUMN_NAME: 'abstract'}, inplace=True) 
else:
    raise KeyError(f"Critical Error: Could not find required abstract column: '{ABSTRACT_COLUMN_NAME}'.")

df_pandas.dropna(subset=['abstract'], inplace=True) 

df = spark.createDataFrame(df_pandas[['abstract']])
df = df.repartition(20) 

print(f"Loaded {df.count()} abstracts.")
print(f"Number of Partitions (Scalable Chunks): {df.rdd.getNumPartitions()}")

# STAGE 2: PARALLEL GENERATIVE INFERENCE (PANDAS UDF PROCESSING)
@pandas_udf("string")
def scalable_summarize(texts: pd.Series) -> pd.Series:
    summarizer = pipeline(
        "summarization",
        model="Falconsai/medical_summarization",
        device=-1 
    )
    summaries = summarizer(
        texts.tolist(),
        max_length=25,
        min_length=15,
        num_beams=4,
        do_sample=False,
        truncation = True
    )
    return pd.Series([s['summary_text'] for s in summaries])

df_summarized = df.withColumn(
    "summary",
    scalable_summarize(col("abstract")) 
)

print("\n--- End-to-End Scalable Pipeline Output (Sample Summaries) ---")
df_summarized.show(5, truncate=False)

# =========================================================================
# NEW: STAGE 2.5: ROUGE VALIDATION QUALITY GATE (This was missing in your V3)
# =========================================================================
print("\n--- Running Distributed ROUGE Validation ---")

@pandas_udf("float")
def calculate_rouge(abstracts: pd.Series, summaries: pd.Series) -> pd.Series:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    # Compare generated summary against the original abstract
    for abstract, summary in zip(abstracts, summaries):
        score = scorer.score(str(abstract), str(summary))['rougeL'].fmeasure
        scores.append(score)
    return pd.Series(scores)

# Add the score to the dataframe
df_scored = df_summarized.withColumn(
    "rouge_score",
    calculate_rouge(col("abstract"), col("summary"))
)

# QUALITY GATE: Filter out summaries with a low ROUGE score (e.g., less than 0.15)
# This prevents AI hallucinations from reaching the Topic Modeler
df_validated = df_scored.filter(col("rouge_score") >= 0.15)

dropped_count = df_scored.count() - df_validated.count()
print(f"Validation Gate: Blocked {dropped_count} low-quality AI summaries from entering the Insight Engine.")

# =========================================================================
# STAGE 3: KNOWLEDGE DISCOVERY (TOPIC MODELING)
# =========================================================================
# CHANGE: Notice we now use 'df_validated' instead of 'df_summarized'
summary_list = df_validated.select("summary").rdd.flatMap(lambda x: x).collect()

stop_words = set(stopwords.words('english'))
processed_docs = []

for text in summary_list:
    tokens = [word.lower() for word in word_tokenize(text)
              if word.isalpha() and word.lower() not in stop_words and len(word) > 2]
    processed_docs.append(tokens)

dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

NUM_TOPICS = 5 
lda_model = LdaModel(
    corpus,
    num_topics=NUM_TOPICS,
    id2word=dictionary,
    passes=15 
)

print("\n--- Final Knowledge Discovery Insights (Integrated Topic Modeling) ---")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic #{idx + 1}: {topic}")


# =========================================================================
# STAGE 4: DATABASE EXPORT
# =========================================================================
print("\n--- STAGE 4: SAVING TO DATABASE ---")

print("Gathering distributed data...")
final_df = df_validated.toPandas()

print("Connecting to local SQL Database (medical_insights.db)...")
db_connection = sqlite3.connect('medical_insights.db')

final_df.to_sql('validated_summaries', db_connection, if_exists='replace', index=False)

print(f"✅ SUCCESS: Saved {len(final_df)} validated records to the database.")
db_connection.close()

spark.stop()