# scalable_pipeline.py: Integrated Scalable Summarization Pipeline

from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from transformers import pipeline
import pandas as pd
import os

# --- PySpark Setup (The Scalable Part) ---
# Initializes Spark Session, the entry point for distributed computing
spark = SparkSession.builder \
    .appName("ScalableAIPipeline") \
    .master("local[*]") \
    .getOrCreate()

# Force partitioning to demonstrate scalability (4 partitions = 4 parallel chunks of work)
spark.conf.set("spark.sql.shuffle.partitions", "4")

# 1. Load the Big Data
# Create a temporary CSV file with sample abstracts
data_content = (
    "abstract\n"
    "Duplications of the alimentary tract are well-known but rare congenital malformations that can occur anywhere in the GI tract. Diagnosis is mainly done via imaging like CT and MRI.\n"
    "A T-cell leukemia variant was identified in 50 patients; further study on the CD8 gene is needed for diagnosis and new therapies.\n"
    "New drug Z-21 shows promise in reversing Alzheimer's symptoms in phase 2 trials. Patient response was positive with minor side effects.\n"
    "Patient was prescribed Metformin 500mg daily for Type 2 Diabetes. Blood sugar levels were closely monitored and dosage was adjusted.\n"
    "The new radiation therapy protocol was applied intravenously to treat a T-cell leukemia variant. Clinical outcomes are pending.\n"
)
data_file_name = "abstracts_to_process.csv"
with open(data_file_name, "w") as f:
    f.write(data_content)

# Read the data into a Spark DataFrame and repartition it for parallel processing
df = spark.read.csv(
    data_file_name,
    header=True,
    inferSchema=True
).repartition(4) # Force 4 partitions

print(f"Number of Partitions (Scalable Chunks): {df.rdd.getNumPartitions()}")

# --- Generative Model Integration (The AI Part via Pandas UDF) ---
# Pandas UDF (User-Defined Function) acts as the technical bridge to run the LLM in parallel.
@pandas_udf("string")
def scalable_summarize(texts: pd.Series) -> pd.Series:
    # This code block runs independently on each of the 4 worker cores/threads.
    
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
# 

# --- Final Output (The Proof) ---
print("\n--- End-to-End Scalable Pipeline Output (Summaries) ---")
df_summarized.show(truncate=False)

# Cleanup and Stop Spark
spark.stop()
try:
    os.remove(data_file_name)
except OSError:
    pass