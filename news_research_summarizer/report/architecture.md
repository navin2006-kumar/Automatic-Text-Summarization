# System architecture and workflow

## Component architecture

```mermaid
flowchart LR
    User[User] --> UI[Streamlit interface]
    UI --> Parse[preprocessing.py<br/>TXT / PDF / DOCX extraction]
    UI --> Core[summarizer.py]
    Parse --> Core
    Core --> Tokenizer[Hugging Face tokenizer]
    Tokenizer --> Split[Token-aware chunking]
    Split --> Model{Selected pretrained model}
    Model --> BART[facebook/bart-large-cnn]
    Model --> T5[t5-base]
    BART --> Reduce[Hierarchical map-reduce]
    T5 --> Reduce
    Reduce --> UI
    UI --> Eval[evaluation.py / Evaluate ROUGE]
    Eval --> Results[results/*.csv and metadata]
```

## Data-flow diagram

```mermaid
flowchart TD
    A[Paste text or upload document] --> B{Input valid?}
    B -- No --> C[Show validation error]
    B -- Yes --> D[Extract and count words]
    D --> E[Select BART or T5]
    E --> F[Tokenizer measures model tokens]
    F --> G[Split at model context boundary]
    G --> H[Generate summary for each chunk]
    H --> I{Intermediate text fits context?}
    I -- No --> J[Summarize intermediate groups recursively]
    J --> I
    I -- Yes --> K[Generate final summary]
    K --> L[Show summary, word counts, time, ratio]
    L --> M[Download text]
```

## Use-case diagram

```mermaid
flowchart LR
    User((User))
    User --> Paste[Paste document text]
    User --> Upload[Upload TXT, PDF, or DOCX]
    User --> Select[Select model and target length]
    User --> Summarize[Generate summary]
    User --> Compare[Compare BART and T5]
    User --> Score[Score against reference with ROUGE]
    User --> Download[Download generated summary]
```

## Workflow notes

The UI delegates parsing to `preprocessing.py`; the same validated source string
then goes to one or both summarizers. Each summarizer uses its own tokenizer and
context limit. Map-reduce covers every token chunk and recursively reduces
intermediate summaries. ROUGE is computed only after generation when a non-empty
human reference is provided. The evaluation CLI instead accepts paired rows and
runs the same rows through both models before writing per-example and aggregate
CSV files.
