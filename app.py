from flask import Flask, request, render_template, url_for
import os

from transformers import pipeline
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load models and keys (consider environment variables)
refinement_model = pipeline("text-generation", model="roberta-base")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
os.environ["SERPER_API_KEY"] = '694df41415d0bf10cbc7e77b5e57ee59cafd38e0'  # Replace with your API key


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search = GoogleSerperAPIWrapper()
    query = request.form['query']

    refined_query = refinement_model(query, num_return_sequences=1)[0]["generated_text"]
    search_results = search.run(refined_query)
    summary = summarization_model(search_results, max_length=128, truncation=True)[0]["summary_text"]

    return render_template('index.html', final_result=summary)

if __name__ == '__main__':
    print("Server Started")
    app.run(debug=True)