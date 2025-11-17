import google.generativeai as genai

# 🔑 Configure with your Gemini API key
genai.configure(api_key="REMOVED")




# Function to test a model
def test_model(model_name):
    print(f"\n🔍 Testing {model_name} ...")
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Summarize the importance of AI in one sentence.")
        print(f"✅ {model_name} is working!")
        print("Response:", response.text)
    except Exception as e:
        print(f"❌ {model_name} failed or key not authorized.")
        print("Error:", e)

# Test both Gemini 2.5 models
test_model("gemini-2.5-flash")
test_model("gemini-2.5-pro")