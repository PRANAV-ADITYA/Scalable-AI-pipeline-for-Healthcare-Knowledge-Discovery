# Day 3: Installation (The Hardest Part!)
# You must install Java first, then Apache Spark (or just use 'pip install pyspark'
# and set environment variables). Follow the steps in the video below.

# Day 4: The Core PySpark Script

from pyspark.sql import SparkSession
import os # We'll use this to manage the Java environment variable

# **CRITICAL STEP:** Create the entry point for Spark
# .master("local[*]") tells Spark to run locally using all your computer's cores.
# This simulates the "cluster" required for distributed computing.
spark = SparkSession.builder \
    .appName("ScalableMedicalDataLoader") \
    .master("local[*]") \
    .getOrCreate()

# 1. Create a sample CSV file to load (Pretend this is your Big Data)
# You can copy and paste this into a file named 'medical_abstracts.csv'
data_file_content = (
    "abstract,category\n"
    "GI tract duplications are rare congenital malformations. Imaging is key for diagnosis.,Diagnosis\n"
    "A T-cell leukemia variant was identified in 50 patients; further study is needed.,Oncology\n"
    "New drug Z-21 shows promise in reversing Alzheimer's symptoms in phase 2 trials.,Neurology"
)
with open("medical_abstracts.csv", "w") as f:
    f.write(data_file_content)

# 2. Load the Big Data file into a Spark DataFrame
# header=True uses the first row for column names.
# inferSchema=True tells Spark to guess the data types.
df = spark.read.csv(
    "medical_abstracts.csv", 
    header=True, 
    inferSchema=True
)
df = df.repartition(4)

# 3. Show the data and the schema
# This is an 'Action' that forces Spark to actually read the data.
print("\n--- Spark DataFrame Output (The Big Data Table) ---")
df.show()

# 4. Print the number of partitions (The "Scalable" Proof)
# Partitions are the chunks of data that Spark distributes to different cores/computers.
print(f"Number of Partitions (Scalable Chunks): {df.rdd.getNumPartitions()}")

# 5. Stop the Spark Session
spark.stop()
# Expected Output: A small table (DataFrame) and a number of partitions greater than 1, proving the data is ready for parallel processing.