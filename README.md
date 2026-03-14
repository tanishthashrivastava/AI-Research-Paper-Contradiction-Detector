# AI Research Paper Contradiction Detector

## Overview

The **AI Research Paper Contradiction Detector** is an AI-powered system designed to analyze research papers and identify contradictions between their conclusions. The project scrapes research papers from arXiv, performs semantic search across papers, and uses Natural Language Processing (NLP) models to detect whether two research findings contradict or support each other.

This tool helps researchers quickly discover conflicting findings across papers and explore related research topics using semantic similarity.

---

## Key Features

### 1. Research Paper Scraping

* Automatically fetches research papers from **arXiv**.
* Extracts titles and summaries to create a dataset.

### 2. Semantic Search

* Uses **sentence embeddings** to find research papers similar to a user query.
* Allows users to explore related research quickly.

### 3. Contradiction Detection

* Uses a **Natural Language Inference (NLI) model** to analyze whether two research conclusions:

  * Contradict each other
  * Support each other
  * Are unrelated

### 4. Automatic Contradiction Analysis

* Automatically checks contradictions across the entire dataset of papers.
* Stores detected contradictions in a CSV file for further analysis.

### 5. Interactive Web Interface

* Built using **Streamlit** for an easy-to-use interface.
* Users can search papers and test contradictions directly from the browser.

---

## Project Architecture

```
User Input
     │
     ▼
Streamlit Web App (app.py)
     │
     ├── Semantic Search
     │      │
     │      ▼
     │ Sentence Transformer Embeddings
     │
     └── Contradiction Detection
            │
            ▼
   Transformer NLI Model
            │
            ▼
      Contradiction Result
```

---

## Project Structure

```
AI-Research-Paper-Contradiction-Detector
│
├── app.py
├── arxiv_scraper.py
├── auto_contradiction_detector.py
├── contradiction_detector.py
├── embeddings.py
├── similarity_search.py
├── requirements.txt
│
├── data
│   ├── papers.csv
│   └── contradictions_found.csv
│
└── README.md
```

---

## Technologies Used

* **Python**
* **Streamlit**
* **Transformers (HuggingFace)**
* **Sentence Transformers**
* **arXiv API**
* **Pandas**
* **Scikit-learn**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Research-Paper-Contradiction-Detector.git
cd AI-Research-Paper-Contradiction-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### 1. Fetch Research Papers

```bash
python arxiv_scraper.py
```

This creates:

```
data/papers.csv
```

---

### 2. Detect Contradictions Across Dataset

```bash
python auto_contradiction_detector.py
```

This generates:

```
data/contradictions_found.csv
```

---

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser:

```
http://localhost:8501
```

---

## Example Usage

### Semantic Search

Input:

```
transformer models for natural language processing
```

Output:

```
Top Similar Papers:
- Title: Transformer for NLP
- Conclusion: Transformer models significantly improve translation accuracy.
```

---

### Contradiction Detection

Paper 1 Conclusion:

```
Transformer models significantly improve machine translation accuracy.
```

Paper 2 Conclusion:

```
Transformer models perform worse than traditional LSTM models in translation tasks.
```

Output:

```
⚠ These papers contradict each other.
Confidence Score: 0.92
```

---

## Future Improvements

* Add larger research datasets
* Improve contradiction detection using fine-tuned models
* Store embeddings in a vector database
* Deploy the application online

---


## License

This project is for academic and research purposes.
