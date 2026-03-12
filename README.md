# RAGAS Evaluation Framework for Customer Ticket Email Threads

This project builds a **RAGAS evaluation dataset** from raw customer support email threads (`.txt` files) and then runs RAGAS metrics on that dataset.

It is tailored to your requirements:
- Extract the **issue** from each thread.
- Rephrase the issue into a natural **question**.
- Extract the **solution** as **ground truth**.
- Rephrase the solution into an **answer**.
- Preserve relevant **context** from the original thread.
- Use **Anthropic models** for extraction/rephrasing.
- Use your **custom embeddings class** for evaluation.

## What gets generated

For each input text file, the dataset row contains:
- `issue_original`
- `question` (rephrased issue)
- `solution_original` (ground truth)
- `ground_truth`
- `answer` (rephrased solution)
- `contexts` (list of context chunks)
- `source_file`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic key:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

## 1) Generate the evaluation dataset from email thread text files

Place your files under a folder, for example:

```text
emails/
  ticket_001.txt
  ticket_002.txt
  ...
```

Then run:

```bash
python rag_eval_pipeline.py generate-dataset \
  --input-dir emails \
  --output-jsonl data/ragas_test_dataset.jsonl \
  --anthropic-model claude-3-5-sonnet-latest
```

This will call Anthropic for each thread and write one JSONL record per file.

## 2) Run RAGAS evaluation using your custom embeddings class

`evaluate` requires a custom embeddings class import path.

Your class should be importable and provide the embeddings interface expected by RAGAS/LangChain (for example methods like `embed_documents` and `embed_query`).

```bash
python rag_eval_pipeline.py evaluate \
  --dataset-jsonl data/ragas_test_dataset.jsonl \
  --anthropic-model claude-3-5-sonnet-latest \
  --embeddings-module my_project.embeddings \
  --embeddings-class MyCustomEmbeddings
```

If your class needs constructor args:

```bash
python rag_eval_pipeline.py evaluate \
  --dataset-jsonl data/ragas_test_dataset.jsonl \
  --anthropic-model claude-3-5-sonnet-latest \
  --embeddings-module my_project.embeddings \
  --embeddings-class MyCustomEmbeddings \
  --embeddings-init-kwargs-json '{"model_name":"my-embed-v1","timeout":30}'
```

This evaluates using RAGAS metrics (defaults):
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `context_precision`
- `context_recall`

You can also choose metrics explicitly:

```bash
python rag_eval_pipeline.py evaluate \
  --dataset-jsonl data/ragas_test_dataset.jsonl \
  --anthropic-model claude-3-5-sonnet-latest \
  --embeddings-module my_project.embeddings \
  --embeddings-class MyCustomEmbeddings \
  --metrics faithfulness,answer_correctness,answer_relevancy
```

## Optional: evaluate your own RAG answers

If you already have model answers (from your RAG system), prepare a JSONL file with:
- `source_file`
- `answer`

Then run:

```bash
python rag_eval_pipeline.py evaluate \
  --dataset-jsonl data/ragas_test_dataset.jsonl \
  --answers-jsonl data/rag_answers.jsonl \
  --anthropic-model claude-3-5-sonnet-latest \
  --embeddings-module my_project.embeddings \
  --embeddings-class MyCustomEmbeddings
```

The script will replace the dataset `answer` with your generated answer and evaluate that output against extracted ground truth.

## Notes

- Anthropic is used for **dataset generation** and evaluator LLM for RAGAS.
- Embeddings are loaded dynamically from your custom class via:
  - `--embeddings-module`
  - `--embeddings-class`
  - optional `--embeddings-init-kwargs-json`
