# Demonstration script

1. Introduce the goal: compare two pretrained sequence-to-sequence models for
   local news and research summarization.
2. Show the project modules and explain that model weights download/cache on
   first inference; point out device selection and model identifiers.
3. Paste a representative news article. Show input words, choose BART, set a
   target length, and summarize. Explain the displayed inference time excludes
   loading/download.
4. Repeat using T5 on the same source. Discuss only observed differences and
   check the outputs against the source.
5. Open the Compare models page and demonstrate paired outputs.
6. Upload a text-based PDF or DOCX and confirm extraction; mention scanned PDFs
   need OCR, which is not included.
7. Demonstrate the ROUGE page with a human reference. Explain that ROUGE measures
   lexical overlap and cannot prove factual correctness.
8. Download a summary and show the resulting text file.
9. If practical, run the CSV evaluation script and show its generated tables;
   otherwise state clearly that benchmark results are pending.
10. Close with limitations: chunk boundaries lose cross-document context,
    generations can be inaccurate, and CPU inference may be slow.

Capture screenshots directly from the running application and save them in
`screenshots/`; include the date and do not edit the screenshot to imply
unperformed behavior.

# Viva questions and answers

**1. What is abstractive summarization?**  
It generates a condensed text sequence rather than only selecting existing
sentences. It can paraphrase, but may also introduce unsupported details.

**2. How do extractive and abstractive methods differ?**  
Extractive methods select source spans; abstractive methods generate new
wording. The application uses abstractive sequence generation.

**3. What is BART?**  
BART is a pretrained denoising sequence-to-sequence Transformer with an encoder
and autoregressive decoder. The project uses the CNN/DailyMail-finetuned model
identifier `facebook/bart-large-cnn`.

**4. What is T5?**  
T5 expresses tasks in a text-to-text format. Here `t5-base` is prompted with
`summarize:` and generates summary text.

**5. Why compare these two models?**  
BART is already fine-tuned for news summarization, while T5 is a general
text-to-text checkpoint. Their behavior is compared experimentally, not assumed
from the model names.

**6. Why is chunking needed?**  
Each model has a finite context window. Token-based chunks keep model inputs
within that window and allow larger inputs to be summarized in stages.

**7. What is map-reduce in this project?**  
The map step summarizes each input chunk. The reduce step combines and
recursively summarizes those intermediate summaries into a final input-sized
representation.

**8. Can chunking preserve all document context?**  
No. Relationships across chunks can be weakened or lost. Numerical findings and
technical claims need checking against the original source.

**9. What does ROUGE measure?**  
ROUGE-1/2 compare n-gram overlap, and ROUGE-L uses longest common subsequence
overlap with a reference. It does not establish correctness or factuality.

**10. How is fairness of the comparison supported?**  
Both models process the same non-empty source/reference rows. The evaluator
records per-example outputs and aggregate metrics; hardware and configuration
must also be reported.

**11. Why are evaluation results pending?**  
No dataset sample/model inference has been run in this workspace. Reporting
placeholder numbers would be misleading; the provided script generates the
measurements when run.

**12. What are the system's limitations?**  
Model hallucinations, finite context, loss at chunk boundaries, extraction
failure for scanned PDFs, hardware-dependent latency, and overlap-only
evaluation.
