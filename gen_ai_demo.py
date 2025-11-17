from transformers import pipeline

# 1. Choose a specialized medical model (Hugging Face Model ID)
# 'Falconsai/medical_summarization' is a T5 model fine-tuned on medical text.
MODEL_NAME = "Falconsai/medical_summarization"

# 2. Create the summarization pipeline - this is the AI engine!
# The pipeline handles loading the model and tokenizer automatically.
summarizer = pipeline("summarization", model=MODEL_NAME)

# 3. Define a long piece of medical text (e.g., a research abstract)
long_medical_text = (
    "Duplications of the alimentary tract are well-known but rare congenital "
    "malformations that can occur anywhere in the gastrointestinal (GI) tract "
    "from the tongue to the anus. While midgut duplications are the most common, "
    "foregut duplications such as esophagus, stomach, and parts 1 and 2 of the "
    "duodenum account for approximately one-third of cases. They are most "
    "commonly seen either in the thorax or abdomen or in both as congenital "
    "thoracoabdominal duplications. Due to the relatively non-specific clinical "
    "signs, diagnosis can only be made confidently using appropriate imaging. "
    "Plain radiographs, ultrasonography (US), or CT scans are sufficient for "
    "diagnosis, but magnetic resonance imaging (MRI) is also very useful."
)

# 4. Generate the summary (with length limits)
# Rerun the script with these new parameters:

summary = summarizer(
    long_medical_text,
    # 1. Significantly reduce the maximum length for a noticeable difference
    max_length=40,
    # 2. Set a minimum length to ensure it's not too short
    min_length=15,
    # 3. Use 'beam search' for better, more human-like generation
    num_beams=4,
    # 4. Turn off sampling for a consistent, high-quality summary
    do_sample=False,
)

# Rerun the script and check the new output!

# 5. Print the result
print("\n--- Original Text ---")
print(long_medical_text)
print("\n--- Generated Summary ---")
print(summary[0]['summary_text'])
# Expected Output: The model generates a new, abstractive summary of the medical text.