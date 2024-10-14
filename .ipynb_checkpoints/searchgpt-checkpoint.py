import os
from langchain.llms import HuggingFaceHub
from langchain.serpapi import SerpAPIWrapper
from transformers import pipeline

# Set your API keys as environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["SERPAPI_API_KEY"] = "7051d1d4aeaffa72ce5aee4001c9eb375cead1212ecc5c24a35ab08f51ffc1b5"

# Load the Hugging Face model (consider using a summarization-specific model)
model_id = "EleutherAI/gpt-neox-20b"  # Or a summarization-focused model like pegasus
model_kwargs = {"temperature": 0.7, "max_new_tokens": 256}
model = HuggingFaceHub(repo_id=model_id, model_kwargs=model_kwargs)

# Create SerpAPIWrapper object
serpapi = SerpAPIWrapper()

# Define a function for query refinement and summarization
def refine_and_summarize(query):
    # Use Hugging Face LLM for query refinement (enabled)
    summarization_pipeline = pipeline("text-generation", model=model_id)
    prompt = "Refine the query to be more concise and relevant for search: " + query
    refined_query = summarization_pipeline(prompt, max_length=64, truncation=True)[0]["generated_text"]

    # Perform real-time search with SerpAPI
    search_results = serpapi.run(query=refined_query or query)["search_results"]

    # Use Hugging Face summarization pipeline (again)
    summary = summarization_pipeline(search_results, max_length=128, truncation=True)[0]["summary_text"]

    return summary

# Example usage
user_query = "What is the latest news about the climate crisis?"
answer = refine_and_summarize(user_query)
print(answer)