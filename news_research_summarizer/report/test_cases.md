# Test cases and evidence record

Run automated tests from the `news_research_summarizer` directory with
`python -m pytest -q`. Record the exact command, date, environment, and output
after execution. The app-specific manual cases below intentionally remain
unverified until tried with installed dependencies and real model weights.

| ID | Test | Expected behavior | Actual behavior / evidence | Status |
|---|---|---|---|---|
| T01 | Normal news article | Non-empty abstractive summary; statistics shown | Pending real model run; no output evidence yet | Pending |
| T02 | Research-paper abstract | Summary retains central question/result | Pending real model run; no output evidence yet | Pending |
| T03 | Long research excerpt | All token chunks processed/reduced or explicit error; no silent truncation | Fake-model chunk coverage in `tests/test_summarizer.py`; actual result pending | Pending |
| T04 | Empty input | Visible validation error; no model call | `test_validate_and_word_count`; execute recorded test command | Pending execution |
| T05 | Very short text | Summary or model limitation/error shown without crash | Fake model verifies short input path; real-model behavior pending | Pending |
| T06 | TXT upload | UTF-8 text extracted and counted | `test_extract_txt`; execute recorded test command | Pending execution |
| T07 | PDF upload | Embedded text extracted; image-only PDF reports no text | Blank PDF extraction checked; text-bearing PDF manual test pending | Pending |
| T08 | DOCX upload | Paragraph/table text extracted | `test_extract_docx`; execute recorded test command | Pending execution |
| T09 | Model-loading failure | Clear visible error; no success-shaped output | Manual test: simulate unavailable model/download and record UI evidence | Pending |
| T10 | BART/T5 comparison | Both models use same source; two summaries/statistics displayed | Fake paired rows test is not real model comparison; manual run pending | Pending |
| T11 | ROUGE with reference | ROUGE-1/2/L and caveat displayed | Fake metric wiring test; real Evaluate metric run pending | Pending |
| T12 | Summary download | Downloaded UTF-8 text matches displayed summary | Manual browser download and file inspection pending | Pending |
| T13 | Unsupported file / empty upload | Useful validation error | `test_reject_unsupported_and_empty_files`; execute recorded test command | Pending execution |

For each manual test, add the observed result, pass/fail, machine/software
details, and a genuine screenshot or output file. Do not change a case to Pass
based solely on its expected behavior.
