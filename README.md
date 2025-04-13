# Legal Text Analyzer

## Data Flow

1. **Frontend (React + Vite)**:
   - The user interacts with the web interface to upload a PDF file.
   - The file is either selected via a file input or dragged and dropped into the upload area.
   - Upon submission, the file is sent to the backend using an HTTP POST request to the `/summarize` endpoint.

2. **Backend (Flask)**:
   - The backend, implemented in `app.py` in the `/model` directory, receives the uploaded PDF file.
   - The file is saved temporarily, and its text content is extracted using the `PyPDF2` library.
   - The extracted text is processed by two summarization models:
     - **BART**: Generates a summary using the `facebook/bart-large-cnn` model.
     - **PEGASUS**: Generates a summary using the `google/pegasus-xsum` model.
   - The summaries are combined using Bayesian Optimization to determine the optimal weights for merging the outputs of the two models.
   - The final combined summary is returned to the frontend as a JSON response.

3. **Frontend (React)**:
   - The frontend receives the summary from the backend and displays it in a scrollable summary box.
   - If an error occurs during processing, an appropriate error message is displayed to the user.

---

## Explanation of `app.py`

The `app.py` file in the `/model` directory serves as the backend for the Legal Text Analyzer application. Below is an explanation of its key components:

### 1. **Dependencies and Initialization**
   - **Flask**: Used to create the web server and handle HTTP requests.
   - **PyPDF2**: Extracts text from uploaded PDF files.
   - **Hugging Face Transformers**: Provides pre-trained models for summarization (BART and PEGASUS).
   - **Bayesian Optimization**: Optimizes the weights for combining summaries from BART and PEGASUS.
   - **Sentence Transformers**: Computes sentence embeddings for evaluating the quality of combined summaries.

### 2. **Endpoints**
   - **`/summarize`**:
     - Accepts a POST request with a PDF file.
     - Validates the file type and extracts text using `PyPDF2`.
     - Generates summaries using BART and PEGASUS models.
     - Combines the summaries using Bayesian Optimization.
     - Returns the final summary along with example citations.

### 3. **Key Functions**
   - **`allowed_file(filename)`**:
     - Checks if the uploaded file has a valid extension (PDF).
   - **`extract_text_from_pdf(pdf_path)`**:
     - Extracts text from the uploaded PDF file.
   - **`evaluate_summary(weight_bart, weight_pegasus, bart_summary, pegasus_summary)`**:
     - Computes a score for the combined summary using cosine similarity of sentence embeddings.
   - **`combine_summaries(bart_summary, pegasus_summary)`**:
     - Uses Bayesian Optimization to find the optimal weights for combining BART and PEGASUS summaries.

### 4. **Summarization Models**
   - **BART**:
     - Pre-trained on CNN/DailyMail dataset for summarization tasks.
   - **PEGASUS**:
     - Pre-trained on XSum dataset for abstractive summarization.

### 5. **Error Handling**
   - Handles errors such as unsupported file types, empty PDFs, and processing failures.
   - Logs errors for debugging purposes and returns appropriate error messages to the frontend.

### 6. **Deployment**
   - The Flask app runs on `http://0.0.0.0:5001` with debugging enabled.
   - Ensures the upload folder exists before starting the server.