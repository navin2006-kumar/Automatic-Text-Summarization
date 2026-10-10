# Automatic Text Summarization of News and Research Papers Using Pre-trained Transformers (BART/T5)

**Student:** [Student name]  
**Degree / department:** B.Tech, Artificial Intelligence and Data Science  
**Institution:** [Institution name]  
**Academic year:** [Academic year]  
**Project guide:** [Guide name]

> **Draft status:** This report describes the source code currently in this
> repository. Experimental results are pending until the evaluator is run against
> a documented dataset sample. Replace placeholders only with measured results and
> institution-approved details.

## Certificate

This is to certify that the project entitled **“Automatic Text Summarization of
News and Research Papers Using Pre-trained Transformers (BART/T5)”** is submitted
by [Student name], under my supervision, in partial fulfilment of the requirements
for the degree of Bachelor of Technology in Artificial Intelligence and Data
Science.

**Guide signature:** __________  **Head of Department:** __________  
**Date:** __________  **Institution seal:** __________

## Declaration

I declare that this project report is my own academic work, except where
acknowledged through citations. Model outputs and experimental metrics must be
reported as generated and must not be represented as verified factual statements.

**Student signature:** __________  **Date:** __________

## Acknowledgement

I express my gratitude to my project guide, department, and institution for their
guidance and support. I also acknowledge the maintainers of the open-source
libraries and pretrained model resources used in this work.

## Abstract

This project implements a local web application for abstractive summarization of
news articles and research-paper text. The application uses the pretrained
`facebook/bart-large-cnn` and `t5-base` sequence-to-sequence models through
Hugging Face Transformers and PyTorch. Inputs can be pasted or extracted from TXT,
PDF, and DOCX files. For inputs beyond the selected model's context, the program
splits text according to tokenizer token counts, generates intermediate
summaries, and recursively reduces them before producing a final summary. The
Streamlit interface provides model selection, target-length controls, model
comparison, word and compression statistics, reference-based ROUGE scoring, and
downloads. An evaluation command processes paired source/reference records using
both models and writes per-sample and aggregate results. **Quantitative
experimental results are pending**; this report does not claim unmeasured scores
or timings. ROUGE is interpreted as text overlap rather than evidence of factual
correctness.

**Keywords:** abstractive summarization, BART, T5, Transformers, long documents,
ROUGE, Streamlit

## Contents

1. Introduction
2. Literature Review
3. System Analysis and Design
4. Methodology
5. Implementation
6. Results and Discussion
7. Conclusion and Future Work
8. References
9. Appendices

## List of figures

1. Component architecture and data flow — `architecture.md`
2. Token-aware map-reduce workflow — `architecture.md`
3. Use-case diagram — `architecture.md`

## List of tables

1. Functional and non-functional requirements
2. Project modules
3. Planned experimental measurements
4. Test cases

# Chapter 1: Introduction

## 1.1 Background

News and research documents can be lengthy, making manual review time-consuming.
Automatic summarization aims to produce a shorter representation that retains
important content. Extractive systems select existing sentences; abstractive
systems generate new text conditioned on the source. Pretrained sequence-to-
sequence Transformers provide a practical basis for abstractive generation
without training a large model from scratch.

## 1.2 Problem statement

The project addresses the need for an accessible local tool that summarizes both
news and research text, supports common document formats, handles inputs larger
than a model context window, and makes model comparison and reference-based
evaluation reproducible.

## 1.3 Motivation

The project offers a demonstrable application of modern NLP, provides a direct
comparison between a summarization-finetuned model and a general text-to-text
model, and makes limitations visible to users rather than presenting generated
text as guaranteed fact.

## 1.4 Objectives

1. Build an abstractive summarization application using pretrained BART and T5.
2. Accept text and TXT/PDF/DOCX documents.
3. Process long inputs through tokenizer-aware hierarchical chunking.
4. Compare outputs, generation time, summary length, and compression.
5. Measure ROUGE-1, ROUGE-2, and ROUGE-L on paired human references.
6. Provide user-facing warnings about context loss and factual verification.

## 1.5 Scope

The implementation is local and English-oriented, has no model training, and
supports documents containing an extractable text layer. It does not include OCR,
source citation extraction, fact verification, or a claim that generated outputs
are suitable without human review. Inference speed and memory use depend on
hardware, input length, and model downloads.

## 1.6 Report organization

Chapter 2 reviews relevant summarization and pretrained Transformer work. Chapter
3 describes requirements and architecture. Chapter 4 explains data, tokenization,
chunking, and evaluation. Chapter 5 describes the application implementation.
Chapter 6 reserves space for measured outcomes and test evidence. Chapter 7
concludes and proposes extensions.

# Chapter 2: Literature Review

## 2.1 Summarization approaches

Extractive systems select sentences or phrases from a document. They tend to
preserve source wording but may produce disconnected selections. Abstractive
systems generate a new sequence and can combine information, but may omit or
misstate facts. The implemented system is abstractive.

## 2.2 Transformer architecture

The Transformer introduced attention-based encoder-decoder processing without
recurrent sequence operations. It forms the architectural basis for both model
families used here [1].

## 2.3 BART

BART is a denoising sequence-to-sequence pretrained model with a bidirectional
encoder and autoregressive decoder. Its paper evaluates generation tasks,
including summarization [2]. This application uses the pretrained
`facebook/bart-large-cnn` model identifier, not a model trained by this project.

## 2.4 T5

T5 frames NLP tasks as text-to-text transformations and studies transfer learning
and pretraining choices [3]. This implementation uses `t5-base` with the
`summarize:` task prefix.

## 2.5 Long-document summarization and evaluation

Long scientific documents introduce discourse and cross-section dependencies.
Cohan et al. study discourse-aware attention for scientific-paper summarization
and describe the arXiv/PubMed domains [4]. This project instead uses a
hierarchical map-reduce procedure suitable for models with limited input
windows; this approximation can lose cross-chunk context. Lin's ROUGE work
defines overlap-oriented automatic measures for summary evaluation [5].

## 2.6 Research gap

This student implementation integrates two existing pretrained models, common
document inputs, long-input chunk orchestration, paired scoring, and a local UI
in one reproducible workflow. It is an engineering demonstration rather than a
new model or a claim of state-of-the-art quality. Its comparison is meaningful
only when both models use the same source/reference sample and recorded settings.

# Chapter 3: System Analysis and Design

## 3.1 Existing system and limitations

Manual summarization requires time and is difficult to reproduce consistently.
Single-model online tools can obscure model configuration, may require sending
documents to external services, and often do not provide a paired local
evaluation. Automatic systems can also hallucinate, truncate long inputs, or
produce results that are difficult to inspect.

## 3.2 Proposed system

The proposed system parses supported local documents, validates non-empty text,
uses the selected model's tokenizer to split the full input, summarizes chunks,
and reduces intermediate summaries to a final context-sized representation.
The Streamlit UI displays word counts, elapsed generation time, compression,
and a downloadable output. Comparison/evaluation runs use both models on the
same source.

## 3.3 Requirements

| Category | Requirement |
|---|---|
| Functional | Accept pasted text and TXT/PDF/DOCX with validation |
| Functional | Select BART or T5 and set target length between 20 and 500 words |
| Functional | Process long text through token-aware hierarchical summarization |
| Functional | Display counts, compression ratio, elapsed time, and download |
| Functional | Compare both models and score against a human reference |
| Non-functional | Run locally on Windows with open-source dependencies |
| Non-functional | Report parsing/model/evaluation failures visibly |
| Non-functional | Do not silently drop source chunks or claim ROUGE proves factuality |

## 3.4 Architecture and data flow

The component architecture, data-flow diagram, use-case diagram, and workflow
are included in `architecture.md`. The primary path is user input → parser and
validation → model tokenizer → map-reduce generation → UI result. Evaluation
adds a human reference → Evaluate ROUGE → result tables/files.

# Chapter 4: Methodology

## 4.1 Dataset selection and protocol

Candidate data sources are CNN/DailyMail for news and the arXiv scientific-paper
dataset described by Cohan et al. [4]. The repository does not contain downloaded
corpora. The experiment must specify exact dataset release/configuration, split,
sample selection, and licensing/terms before execution. A manageable first
experiment can use a fixed sample of 20 test records per domain, but those
records must be selected reproducibly, stored locally as a CSV with `text` and
`reference` columns, and passed identically to both models. The model scores
remain **pending** until this is executed.

## 4.2 Preprocessing and tokenization

TXT is decoded as UTF-8 (including BOM), PDF pages are extracted using pypdf,
and DOCX paragraphs and tables are extracted using python-docx. Blank or
unsupported documents generate explicit errors. Word counts use whitespace
splitting. Model input lengths are determined by each selected model tokenizer;
no fixed character count approximates token length.

## 4.3 Models and generation

The model identifiers are `facebook/bart-large-cnn` and `t5-base`. Models load
lazily, use CUDA when PyTorch reports it available, otherwise CPU. Generation
uses beam search (`num_beams=4`), a no-repeat trigram constraint, and a maximum
new-token setting derived from the configured target word count. Model weights
are not bundled and are downloaded/cached by Transformers on first use.

## 4.4 Long-document processing

The tokenizer partitions source tokens under the effective model input limit;
the T5 task prefix is accounted for in that limit. Each initial chunk receives a
map summary. If joined intermediate summaries exceed context, they are divided
and reduced recursively. A reduction that fails to shrink or exceeds eight
passes raises an error rather than silently omitting chunks. The final
context-fitting text is summarized with the requested generation limit.
Chunking may still lose discourse relationships and repeated/contradictory
details.

## 4.5 Metrics

ROUGE-1 and ROUGE-2 measure unigram and bigram overlap, while ROUGE-L uses a
longest-common-subsequence basis [5]. The implementation uses `evaluate` and
stemming. Inference time is measured around the `summarize` call; model loading
and first-time downloading are excluded. Compression ratio is summary word
count divided by source word count. These metrics do not establish factual
correctness or semantic adequacy.

# Chapter 5: Implementation

## 5.1 Hardware and software

The intended environment is a Windows laptop with 16 GB system RAM, Python 3.11,
PyTorch, Transformers, Streamlit, Pandas, NumPy, Evaluate, pypdf, and
python-docx. The actual machine CPU/GPU, operating system build, package versions,
and PyTorch device must be recorded during experimentation; they are not
asserted here. CPU execution is supported but can be slow. Large model memory
requirements vary by framework build and inference settings.

## 5.2 Module descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, workflows, result display, download |
| `preprocessing.py` | Input validation, TXT/PDF/DOCX extraction, word count |
| `summarizer.py` | Pretrained model loading, token-aware map-reduce generation |
| `evaluation.py` | Paired BART/T5 experiment and ROUGE calculation/output |
| `requirements.txt` | Runtime and test dependency declarations |
| `tests/` | Unit tests, including fake model/metric tests |
| `data/` | Local user-selected dataset inputs |
| `results/` | Generated experiment outputs |
| `screenshots/` | Genuine application screenshots captured after launch |

## 5.3 Interface and error handling

The Streamlit interface has Summarize, Compare models, and ROUGE evaluation
pages. It supports file uploads, pasted text, model selection, target word
length, loading indicators, visible errors, and summary downloads. Failure to
parse input, load dependencies/weights, generate, or score is reported instead
of silently returning a successful result.

# Chapter 6: Results and Discussion

## 6.1 Experimental setup

| Field | Value |
|---|---|
| Dataset/configuration and license | Pending dataset selection |
| Test samples | Pending |
| Hardware/software versions | Pending local measurement |
| Model identifiers | `facebook/bart-large-cnn`; `t5-base` |
| Input processing | Tokenizer-sized map-reduce chunks |
| Generation target | 150 words by default; configurable 20–500 |

## 6.2 Quantitative results

**Status: PENDING — no model evaluation has been executed in this workspace.**
Do not fill this table with estimates. Run `evaluation.py` on the selected
paired CSV and transcribe values from its output files.

| Model | Samples | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean seconds | Mean summary words | Mean compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| BART | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| T5 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

## 6.3 Qualitative analysis

No generated model summaries are included as observations because pretrained
model inference has not yet been run here. After running the application,
select at least one news and one research example and discuss preservation of
central claims, figures, technical terms, omissions, and unsupported content.
Compare both outputs against the same source and reference.

## 6.4 Test cases and status

Unit tests verify parser behavior, empty/unsupported input checks, chunk
orchestration using a fake model, paired row selection, and metric wiring using
a fake ROUGE implementation. These are not substitutes for real model
inference. The full manual app matrix remains **pending** until run on the
student's machine; complete `test_cases.md` with actual observations.

## 6.5 Limitations

The selected models have finite context and can omit cross-chunk relationships.
PDF extraction cannot read image-only/scanned pages. Generation can hallucinate
or distort numerical details. ROUGE measures overlap only, timing varies with
hardware and model cache state, and the current evaluation utility expects a
prepared CSV rather than downloading a dataset itself.

# Chapter 7: Conclusion and Future Work

The repository implements the central application pipeline: local document
extraction, two pretrained summarizers, token-aware long-document processing,
Streamlit workflows, paired evaluation tooling, and tests. The goals of
installable integration and reproducible evaluation tooling are represented in
code; the quality and speed comparison objective cannot be concluded until real
experiments run. Future work can add dataset acquisition scripts with pinned
versions, OCR, section-aware chunking, factual consistency evaluation,
human-rating protocols, additional scientific summarizers, and memory/performance
optimizations.

# References

[1] A. Vaswani et al., “Attention Is All You Need,” in *Advances in Neural
Information Processing Systems*, vol. 30, 2017.

[2] M. Lewis et al., “BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension,” in *Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics*,
2020, pp. 7871–7880, doi: 10.18653/v1/2020.acl-main.703.

[3] C. Raffel et al., “Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer,” *Journal of Machine Learning Research*, vol. 21,
no. 140, pp. 1–67, 2020.

[4] A. Cohan, F. Dernoncourt, D. S. Kim, T. Bui, S. Kim, W. Chang, and
N. Goharian, “A Discourse-Aware Attention Model for Abstractive Summarization
of Long Documents,” in *Proceedings of NAACL-HLT 2018*, vol. 2, 2018,
pp. 615–621, doi: 10.18653/v1/N18-2097.

[5] C.-Y. Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,” in
*Text Summarization Branches Out*, 2004, pp. 74–81.

[6] Hugging Face, “facebook/bart-large-cnn model card.” [Online]. Available:
https://huggingface.co/facebook/bart-large-cnn. Accessed: 10-Oct-2026.

[7] Hugging Face, “t5-base model card.” [Online]. Available:
https://huggingface.co/google-t5/t5-base. Accessed: 10-Oct-2026.

# Appendices

## Appendix A: Installation and run commands

From the `news_research_summarizer` folder on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Appendix B: Test evidence

Run `python -m pytest -q` and preserve its actual terminal output for the report.
Manual model-loading, application, download, and reference evaluation evidence
must be captured from the local run; no screenshot is included until captured.
