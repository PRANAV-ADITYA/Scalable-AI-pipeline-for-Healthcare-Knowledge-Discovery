# full_pipeline.py: The COMPLETE End-to-End Scalable AI Pipeline (V3: Final Integrated Version)

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from transformers import pipeline
import pandas as pd
import os 
import nltk
from datasets import load_dataset 

# --- Imports for Knowledge Discovery (Topic Modeling) ---
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- NLTK Downloads (Ensuring Topic Modeling Resources Exist) ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK resources: stopwords and punkt...")
    nltk.download('stopwords')
    nltk.download('punkt')
    print("NLTK downloads complete.")

# --- PySpark Setup (The Scalable Part) ---
spark = SparkSession.builder \
    .appName("FullScalableAIPipeline") \
    .master("local[*]") \
    .getOrCreate()

# Force partitioning to demonstrate scalability over a large dataset
spark.conf.set("spark.sql.shuffle.partitions", "20") 

# 1. Load the Big Data (Thousands of Abstracts from Hugging Face)
print("\n--- Loading 5000 real biomedical abstracts from PubMed ---")

dataset = load_dataset("cyrilzakka/pubmed-medline", split='train[:5000]')
df_pandas = dataset.to_pandas()

# --- FIX START: Renaming the correct column to 'abstract' ---
# The abstract text is in the 'content' column based on previous debugging.
ABSTRACT_COLUMN_NAME = 'content' 

# Prepare the Pandas DataFrame
if ABSTRACT_COLUMN_NAME in df_pandas.columns:
    # Rename the column 'content' to 'abstract' so the rest of the script works
    df_pandas.rename(columns={ABSTRACT_COLUMN_NAME: 'abstract'}, inplace=True) 
else:
    # Critical error if the column is still not found
    raise KeyError(f"Critical Error: Could not find required abstract column: '{ABSTRACT_COLUMN_NAME}'.")

# Drop rows where the abstract text is missing (ensuring clean data)
df_pandas.dropna(subset=['abstract'], inplace=True) 
# --- FIX END ---

# Convert the large Pandas DataFrame into a Spark DataFrame for distributed processing
df = spark.createDataFrame(df_pandas[['abstract']])

# Repartition the DataFrame to demonstrate parallel processing over a large chunk of data
df = df.repartition(20) 

print(f"Loaded {df.count()} abstracts.")
print(f"Number of Partitions (Scalable Chunks): {df.rdd.getNumPartitions()}")

# --- Generative Model Integration (The AI Part via Pandas UDF) ---
# Pandas UDF acts as the technical bridge to run the LLM in parallel on Spark's workers.
@pandas_udf("string")
def scalable_summarize(texts: pd.Series) -> pd.Series:
    # Load the AI Model locally on the worker node
    summarizer = pipeline(
        "summarization",
        model="Falconsai/medical_summarization",
        device=-1 # Use CPU on the worker (reliable)
    )

    # Configure the generation parameters
    summaries = summarizer(
        texts.tolist(),
        max_length=40,
        min_length=15,
        num_beams=4,
        do_sample=False
    )
    
    # Return the summary text back to Spark
    return pd.Series([s['summary_text'] for s in summaries])

# Apply the UDF to the DataFrame
df_summarized = df.withColumn(
    "summary",
    scalable_summarize(col("abstract")) # This is where parallel summarization happens
)

# Display a sample of the Scalable Summarization result (first 5 rows)
print("\n--- End-to-End Scalable Pipeline Output (Sample Summaries) ---")
df_summarized.show(5, truncate=False)

# --- Knowledge Discovery Integration (Topic Modeling) ---

# 1. Collect Summaries from Spark (Needed for NLTK/Gensim)
# WARNING: Collecting 5000 documents can take a moment.
summary_list = df_summarized.select("summary").rdd.flatMap(lambda x: x).collect()

# 2. Preprocessing for Topic Modeling
stop_words = set(stopwords.words('english'))
processed_docs = []

for text in summary_list:
    tokens = [word.lower() for word in word_tokenize(text)
              if word.isalpha() and word.lower() not in stop_words and len(word) > 2]
    processed_docs.append(tokens)

# 3. Create Dictionary and Corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# 4. Run the LDA Topic Model
NUM_TOPICS = 5 
lda_model = LdaModel(
    corpus,
    num_topics=NUM_TOPICS,
    id2word=dictionary,
    passes=15 
)

# 5. Display Final Insight
print("\n--- Final Knowledge Discovery Insights (Integrated Topic Modeling) ---")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic #{idx + 1}: {topic}")

# Cleanup and Stop Spark
spark.stop()