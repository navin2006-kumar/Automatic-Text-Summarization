# News and Research Paper Summarizer (BART/T5)

A local Streamlit application for abstractive summaries using pretrained
`facebook/bart-large-cnn` and `t5-base`. It accepts pasted text and TXT/PDF/DOCX
documents, tokenizes the full document into model-sized chunks, and uses map-reduce
summarization for long inputs. It also provides paired model comparison and
reference-based ROUGE evaluation.

> **Hardware note:** BART-large and T5-base are substantial models. CPU inference is
> supported but may be slow; CUDA is selected automatically when available. Weights
> are downloaded from Hugging Face on first use and cached locally. No model is
> included in this repository. First test with a small input.

## Setup on Windows

Open PowerShell in this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then
`.\.venv\Scripts\python.exe -m streamlit run app.py`. Install a PyTorch build
appropriate to the laptop (CPU or CUDA) using the official PyTorch selector if the
default pip build is unsuitable.

Run unit tests from this directory:

```powershell
python -m pytest -q
```

The unit tests use test doubles for model inference; they do not download pretrained
weights and therefore do not constitute quality benchmarks.

## Reproducible paired evaluation

Prepare a UTF-8 CSV with one source and one human reference summary per row:

```csv
text,reference
"Full source document text...","Human-written reference summary..."
```

Then run:

```powershell
python evaluation.py --input data\evaluation.csv --output-dir results --max-summary-words 150
```

Both models run on the same non-empty rows. `results\evaluation_details.csv` stores
per-sample summaries, ROUGE values, timings, and compression ratios;
`results\evaluation_summary.csv` stores aggregate values. Model initialization and
first-run download are excluded from each timed summarization call. Record model
versions, device, dataset version, and sample selection alongside results before
making academic claims. Dataset names/sources and the current **pending** status of
the experiment are documented in `report\project_report.md`.

ROUGE-1/2/L are lexical-overlap metrics. They do not demonstrate factuality,
faithfulness, or human preference. Use the application as decision support and
verify output against its source.

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit input, summaries, model comparison, evaluation and downloads |
| `summarizer.py` | Lazy model loading, token-aware chunking, inference, statistics |
| `preprocessing.py` | TXT/PDF/DOCX extraction, input validation and word counting |
| `evaluation.py` | Paired BART/T5 ROUGE experiment and CSV/JSON output |
| `requirements.txt` | Python runtime and test dependencies |
| `data/` | User-prepared datasets; not populated with large copyrighted corpora |
| `results/` | Generated evaluation outputs; no unmeasured results are checked in |
| `screenshots/` | Genuine captures from a locally running app |
| `report/project_report.md` | Formal report draft aligned to the implemented system |
| `report/architecture.md` | Architecture and workflow diagrams |
| `tests/` | Unit tests for parsers, validation, chunk orchestration and evaluation |

## Dataset options and experiment protocol

Candidate paired datasets are CNN/DailyMail (news) and arXiv (scientific papers).
For example, CNN/DailyMail is distributed through Hugging Face Datasets as
`cnn_dailymail` (choose and record a specific configuration/version); the arXiv
corpus is described by Cohan et al. (2018). Do not silently mix dataset splits or
remove long records without recording the selection. Create a manageable,
deterministic sample (e.g., 20 test examples from each selected split) and retain
the same source/reference rows for both models. Dataset acquisition and model
inference can require significant disk, RAM, network, and CPU/GPU time.

The current repository does not claim completed benchmark values: run the evaluator
on your selected data and report its generated CSV values with the machine/model
metadata. The report labels results as pending until then.

## Long-document behavior and limitations

Input is split by the selected tokenizer, not by character count, and all token
segments are passed through map-reduce summarization. Intermediate summaries are
reduced to fit the model context window. The code raises an explicit error if
reduction fails rather than silently discarding remaining chunks. Chunking can
weaken document-wide context, merge unrelated claims, or omit relations between
sections. Important numbers, terminology, and claims must be checked against the
original document. Scanned PDFs without an embedded text layer require OCR, which
is intentionally not bundled.

## Troubleshooting

- **First model run is slow/fails:** verify network access to Hugging Face, free disk
  space, and that the selected PyTorch build is installed.
- **Out of memory:** close other applications, use CPU PyTorch or a smaller model,
  or reduce concurrent workloads. The app evaluates models sequentially.
- **Scanned PDF produces no text:** OCR is outside this implementation; use a
  text-based PDF or extract text with an approved OCR tool before upload.
- **ROUGE package/network error:** Evaluate may need to download its ROUGE metric
  script/resources the first time; rerun only after verifying the cause.
- **No factuality guarantee:** inspect summary claims, figures, and units in source.

## Academic materials

See `report\project_report.md` for the formal report draft, `report\architecture.md`
for diagrams, and the test files for executable checks. No screenshots or benchmark
results are fabricated; capture screenshots and fill the results section only after
running the app/evaluation on the target laptop.
