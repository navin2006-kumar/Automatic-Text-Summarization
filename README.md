# 📰 Automatic Text Summarization of News and Research Papers Using BART/T5

An AI-powered text summarization application that generates concise and meaningful summaries of news articles and research papers using pre-trained Transformer models, **BART** and **T5**.

Built with Python, Hugging Face Transformers, PyTorch, and Streamlit, this project aims to make lengthy documents easier to understand by extracting their key information through abstractive text summarization.

## 📌 Project Overview

With the rapid growth of digital news and scientific publications, reading and understanding lengthy documents can be time-consuming. This project provides an automated solution that uses pre-trained deep learning models to generate shorter summaries while attempting to preserve the essential meaning of the original text.

The application allows users to input text, select a summarization model, generate summaries, and compare model performance through a simple web interface.

## 🎯 Objectives

- Develop an automatic text summarization system using pre-trained Transformer models.
- Summarize news articles and research-paper text.
- Implement and compare BART and T5.
- Support long documents using chunk-based summarization.
- Evaluate summary quality using ROUGE metrics.
- Measure inference time and summary compression.
- Provide an interactive web interface using Streamlit.

## ✨ Features

- **Dual Transformer Models:** Choose between BART and T5.
- **Text Input:** Paste news articles, abstracts, or research-paper sections.
- **Document Upload:** Support for TXT, PDF, and DOCX files.
- **Automatic Summarization:** Generate concise, readable summaries.
- **Long-Document Handling:** Split lengthy inputs into manageable chunks and summarize the intermediate results.
- **Model Comparison:** Compare BART and T5 on the same input.
- **Summary Statistics:** Display original word count, summary word count, compression ratio, and inference time.
- **ROUGE Evaluation:** Evaluate generated summaries against human-written reference summaries.
- **Download Summary:** Save generated summaries as text files.
- **Local Execution:** Run the application on your own computer without requiring a paid AI API.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Loading and running pre-trained models |
| BART | Abstractive summarization, primarily for news |
| T5 | General text-to-text summarization |
| Streamlit | Interactive web application |
| Pandas | Organizing evaluation results |
| Evaluate | Calculating evaluation metrics |
| ROUGE | Measuring overlap with reference summaries |
| pypdf | Extracting text from PDF files |
| python-docx | Extracting text from DOCX files |

## 🧠 Models Used

### 1. BART

**Model:** `facebook/bart-large-cnn`

BART is a denoising sequence-to-sequence Transformer model fine-tuned on the CNN/DailyMail dataset for summarization. It is a useful starting point for generating summaries of news articles.

### 2. T5

**Model:** `t5-base`

T5 is a text-to-text Transformer model that converts input text into output text. It can perform summarization when provided with the appropriate task prefix and generation settings.

The project compares both models using common test inputs. Their actual performance depends on the document type, input length, and evaluation dataset.

## 🏗️ System Architecture

```text
            User Input
                 |
                 v
       Text / Document Upload
                 |
                 v
       Text Extraction and
          Preprocessing
                 |
                 v
        Tokenization and
       Long-Text Chunking
                 |
          +------+------+
          |             |
          v             v
       BART Model    T5 Model
          |             |
          v             v
       Summary       Summary
          |             |
          +------+------+
                 |
                 v
       Model Comparison
       and Evaluation
                 |
                 v
       Streamlit Results
```

## 📂 Project Structure

```text
news-research-text-summarizer/
│
├── app.py
├── summarizer.py
├── preprocessing.py
├── evaluation.py
├── requirements.txt
├── README.md
│
├── data/
│   └── README.md
│
├── results/
│   └── README.md
│
├── screenshots/
│   └── README.md
│
├── report/
│   └── PROJECT_REPORT_TEMPLATE.md
│
└── tests/
    ├── README.md
    └── test_preprocessing.py
```

**File descriptions:**

- `app.py` — Streamlit application and user interface.
- `summarizer.py` — BART and T5 model loading, summarization, and comparison.
- `preprocessing.py` — Text cleaning, word counting, and document extraction.
- `evaluation.py` — ROUGE evaluation and compression-ratio calculations.
- `requirements.txt` — Python dependencies.
- `data/` — Sample texts and permitted evaluation datasets.
- `results/` — Experimental outputs and evaluation results.
- `screenshots/` — Screenshots captured from the running application.
- `report/` — Academic project report template.
- `tests/` — Automated utility tests.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or 3.11 recommended.
- Git.
- Visual Studio Code or another Python IDE.
- Internet connection for the initial model download.
- Approximately 16 GB RAM recommended for local CPU inference.
- Several gigabytes of free disk space for model weights and dependencies.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/news-research-text-summarizer.git
cd news-research-text-summarizer
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If you use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser to use the application.

**Note:** The first run downloads the selected model weights from Hugging Face. Model loading and summarization may take several minutes on a CPU.

## 🚀 How to Use

1. Launch the Streamlit application.
2. Open the Summarize tab.
3. Paste an article or upload a TXT, PDF, or DOCX file.
4. Select BART or T5.
5. Configure the desired summary length.
6. Click **Generate summary**.
7. Review the generated summary and its statistics.
8. Use the comparison tab to run both models on the same input.
9. Provide a human-written reference summary in the evaluation tab to calculate ROUGE scores.
10. Download the generated summary if required.

## 📊 Evaluation Metrics

### ROUGE-1
Measures unigram overlap between the generated summary and the reference summary.

### ROUGE-2
Measures bigram overlap between the generated summary and the reference summary.

### ROUGE-L
Measures overlap based on the longest common subsequence.

### Inference Time
Measures the time taken to generate a summary under the tested hardware and software conditions.

### Compression Ratio

The percentage reduction in word count is calculated as:

```text
Compression Ratio (%) =
((Original Word Count - Summary Word Count)
 / Original Word Count) × 100
```

ROUGE scores require reference summaries. Use the same test examples for both models and record actual results rather than assuming one model performs better.

## 🧪 Testing

Run the available unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

Additional manual tests should cover:

- News article summarization.
- Research-paper abstract summarization.
- Long-document summarization.
- Empty and very short inputs.
- TXT, PDF, and DOCX uploads.
- Model loading and inference errors.
- BART and T5 comparison.
- ROUGE evaluation with reference summaries.

Record the actual outcomes and screenshots after running each test.

## ⚠️ Limitations

- Generated summaries may omit important details or contain factual errors.
- BART-large-CNN is primarily optimized for news summarization.
- T5-base is a general text-to-text model and may require suitable generation settings.
- Long-document chunking may lose context between sections.
- PDF text extraction may not preserve tables, figures, equations, or multi-column layouts.
- CPU inference may be slow for lengthy documents.
- ROUGE measures text overlap and does not independently verify factual correctness.
- Model performance must be evaluated on appropriate news and research-paper samples.

Always verify important facts, statistics, and scientific conclusions against the original document.

## 🔮 Future Enhancements

- Fine-tune models on scientific-paper datasets.
- Add multilingual summarization.
- Implement section-aware research-paper summarization.
- Improve factual consistency checking.
- Add citation-aware summaries.
- Support additional document formats.
- Develop a larger evaluation dataset.
- Add human evaluation of summary quality.
- Deploy the application to a suitable hosting platform.

## 🎓 Academic Project

**Project Type:** Capstone Project  
**Domain:** Artificial Intelligence, Deep Learning, and Natural Language Processing  
**Primary Models:** BART and T5  
**Interface:** Streamlit

This project demonstrates the use of pre-trained Transformer models for abstractive text summarization and comparative evaluation.

## 📚 References

1. M. Lewis et al., “BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension,” *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 2020.
2. C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” *Journal of Machine Learning Research*, vol. 21, no. 140, pp. 1–67, 2020.
3. C.-Y. Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,” *Text Summarization Branches Out*, 2004.
4. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers
5. Streamlit Documentation: https://docs.streamlit.io/

## 👨‍💻 Author

**Navinkumar M**

B.Tech Artificial Intelligence and Data Science

GitHub: https://github.com/navin2006-kumar

---

If you find this project useful, consider giving the repository a ⭐.
