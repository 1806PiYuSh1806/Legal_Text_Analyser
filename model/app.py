import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import logging

from bayes_opt import BayesianOptimization
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
# Hugging Face transformers
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Setup logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Initialize summarization pipelines for legal text analysis.
# These models can produce longer summaries; adjust parameters as needed.
app.logger.info("Loading BART summarizer model...")
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
app.logger.info("Loading PEGASUS summarizer model...")
pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum")

def evaluate_summary(weight_bart, weight_pegasus, bart_summary, pegasus_summary):
    """
    Scoring function to evaluate the quality of a combined summary
    using cosine similarity of sentence embeddings.
    """
    combined = (
        weight_bart * np.array(model.encode(bart_summary)) +
        weight_pegasus * np.array(model.encode(pegasus_summary))
    )
    
    # We simulate a pseudo-reference using a simple average of both summaries
    reference = model.encode(bart_summary + " " + pegasus_summary)
    
    score = cosine_similarity([combined], [reference])[0][0]
    return score

def combine_summaries(bart_summary, pegasus_summary):
    """
    Combine summaries from BART and PEGASUS using Bayesian Optimization
    to determine optimal weights.
    """

    # Define the objective function for Bayesian Optimization
    def objective(weight_bart):
        weight_pegasus = 1.0 - weight_bart
        return evaluate_summary(weight_bart, weight_pegasus, bart_summary, pegasus_summary)

    # Run Bayesian Optimization to find best weight for BART
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"weight_bart": (0.0, 1.0)},
        verbose=0,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=10)

    # Get best weight
    best_weight_bart = optimizer.max["params"]["weight_bart"]
    best_weight_pegasus = 1.0 - best_weight_bart

    # Combine summaries using the optimized weights
    bart_embedding = np.array(model.encode(bart_summary))
    pegasus_embedding = np.array(model.encode(pegasus_summary))
    combined_embedding = best_weight_bart * bart_embedding + best_weight_pegasus * pegasus_embedding

    # Decode to string not possible directly, so concatenate text for now
    combined_summary = (
        f"Optimized Weights -> BART: {best_weight_bart:.2f}, PEGASUS: {best_weight_pegasus:.2f}\n\n"
        "BART Summary:\n" + bart_summary + "\n\n" +
        "PEGASUS Summary:\n" + pegasus_summary
    )

    return combined_summary

@app.route('/summarize', methods=['POST'])
def summarize():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        try:
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(temp_path)
            if not extracted_text.strip():
                return jsonify({"error": "No text could be extracted from the PDF."}), 400

            # Generate summary using BART
            bart_output = bart_summarizer(
                extracted_text, max_length=300, min_length=100, do_sample=False
            )
            bart_summary = bart_output[0]['summary_text']

            # Generate summary using PEGASUS
            pegasus_output = pegasus_summarizer(
                extracted_text, max_length=300, min_length=100, do_sample=False
            )
            pegasus_summary = pegasus_output[0]['summary_text']

            # Combine the two summaries (simulate Bayesian optimization for weight tuning)
            combined_summary = combine_summaries(bart_summary, pegasus_summary)

            # Append sample citations; in a production system, you might extract real citations.
            citations = (
                "\n\nCitations:\n"
                "1. Example Legal Citation: Roe v. Wade, 410 U.S. 113 (1973).\n"
                "2. Example Legal Citation: Brown v. Board of Education, 347 U.S. 483 (1954)."
            )
            final_summary = combined_summary + citations

            return jsonify({"summary": final_summary})
        except Exception as e:
            app.logger.error("Error during processing: " + str(e))
            return jsonify({"error": "Error processing the PDF."}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        return jsonify({"error": "Unsupported file type. Only PDF files are allowed."}), 400

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5001, debug=True)