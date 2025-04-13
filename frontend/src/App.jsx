import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // Handle file selection via input
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // Drag & drop event handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  // Submit the form: sends PDF to backend for legal text analysis
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setIsLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error uploading file:', error);
      setSummary('Error processing the file.');
    } finally {
      setIsLoading(false);
    }
  };

  // Analysis is considered started if we are loading or the summary exists.
  const analysisStarted = isLoading || summary;

  // Reusable upload box element.
  const uploadBox = (
    <form
      className="bg-gray-800 rounded-lg shadow-lg p-6 transition-all duration-1500"
      onSubmit={handleSubmit}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div
        className={`border-2 border-dashed rounded-md p-6 text-center transition-colors duration-300 ${
          dragActive ? 'border-blue-400 bg-gray-700' : 'border-gray-700'
        }`}
      >
        <input
          type="file"
          accept=".pdf"
          className="hidden"
          id="fileInput"
          onChange={handleFileChange}
        />
        {file ? (
          <p className="text-gray-300">Selected file: {file.name}</p>
        ) : (
          <label htmlFor="fileInput" className="cursor-pointer">
            <p className="text-gray-300">
              Drag and drop a PDF file here or click to select one
            </p>
          </label>
        )}
      </div>
      <button
        type="submit"
        className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded transition-colors duration-300"
        disabled={isLoading}
      >
        {isLoading ? 'Analyzing...' : 'Analyze'}
      </button>
    </form>
  );

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 p-4">
      <h1 className="text-4xl font-bold text-gray-200 mb-8">Legal Text Analyzer</h1>
      {!analysisStarted ? (
        // Initial state: centered upload box.
        <div className="w-full max-w-md">{uploadBox}</div>
      ) : (
        // Analysis state: two-column layout.
        <div className="flex flex-row items-start justify-center gap-8 w-full max-w-4xl">
          {/* Upload box container: animates width and position */}
          <div
            className="transition-all duration-1500"
            style={{
              width: analysisStarted ? '30%' : '100%',
              transform: analysisStarted ? 'translateX(-20%)' : 'translateX(0)',
            }}
          >
            {uploadBox}
          </div>
          {/* Summary box: larger and scrollable */}
          <div className="w-2/3 transition-all duration-1500">
            {isLoading ? (
              <div className="bg-gray-800 rounded-lg shadow-lg p-6 h-80 flex items-center justify-center">
                <p className="text-gray-400">Analyzing...</p>
              </div>
            ) : (
              <div className="bg-gray-800 rounded-lg shadow-lg p-6 h-80 overflow-y-auto">
                <h2 className="text-2xl font-semibold text-gray-200 mb-4">Summary</h2>
                <p className="text-gray-300 whitespace-pre-line">{summary}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
