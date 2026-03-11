import argparse
import importlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from anthropic import Anthropic
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


SYSTEM_PROMPT = """
You are an expert support quality analyst.
Given a customer support email/ticket thread, extract structured data.
Return ONLY valid JSON with the exact schema:
{
  "issue_original": "string",
  "question": "string",
  "solution_original": "string",
  "ground_truth": "string",
  "answer": "string",
  "contexts": ["string", "string"],
  "confidence": 0.0
}

Rules:
- issue_original: concise extraction of the core customer problem from the thread.
- question: rephrase the issue as a clean user question.
- solution_original: concise extraction of the actual resolution in the thread.
- ground_truth: normalized version of solution_original for evaluation.
- answer: rephrase the solution naturally as if an assistant answered the question.
- contexts: 2-6 short context chunks copied/paraphrased from thread that support the solution.
- confidence: 0 to 1 quality confidence.
- If no clear solution exists, set solution_original/ground_truth/answer to "NOT_AVAILABLE" and keep contexts relevant.
""".strip()


@dataclass
class TicketRecord:
    source_file: str
    issue_original: str
    question: str
    solution_original: str
    ground_truth: str
    answer: str
    contexts: List[str]
    confidence: float


def read_text_files(input_dir: Path) -> Dict[str, str]:
    files = sorted(input_dir.glob("*.txt"))
    data: Dict[str, str] = {}
    for file in files:
        data[file.name] = file.read_text(encoding="utf-8", errors="ignore")
    return data


def _extract_json_block(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def anthropic_extract(
    client: Anthropic,
    model: str,
    thread_text: str,
    max_tokens: int = 1200,
) -> dict:
    user_prompt = f"Thread:\n\n{thread_text}\n\nReturn JSON only."
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = "".join(
        block.text for block in msg.content if hasattr(block, "text")
    )
    parsed = json.loads(_extract_json_block(response_text))
    return parsed


def build_dataset(input_dir: Path, output_jsonl: Path, anthropic_model: str) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    client = Anthropic(api_key=api_key)
    raw_threads = read_text_files(input_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for source_file, thread_text in raw_threads.items():
            extracted = anthropic_extract(client, anthropic_model, thread_text)
            record = TicketRecord(
                source_file=source_file,
                issue_original=extracted.get("issue_original", "").strip(),
                question=extracted.get("question", "").strip(),
                solution_original=extracted.get("solution_original", "").strip(),
                ground_truth=extracted.get("ground_truth", "").strip(),
                answer=extracted.get("answer", "").strip(),
                contexts=extracted.get("contexts", []) or [],
                confidence=float(extracted.get("confidence", 0.0) or 0.0),
            )
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    print(f"Wrote dataset to {output_jsonl}")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_answers(dataset_rows: List[dict], answers_rows: List[dict]) -> List[dict]:
    by_source = {r["source_file"]: r["answer"] for r in answers_rows if "source_file" in r and "answer" in r}
    merged = []
    for row in dataset_rows:
        row = dict(row)
        if row.get("source_file") in by_source:
            row["answer"] = by_source[row["source_file"]]
        merged.append(row)
    return merged


def load_custom_embeddings(module_path: str, class_name: str, init_kwargs_json: Optional[str]):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    init_kwargs = json.loads(init_kwargs_json) if init_kwargs_json else {}
    if not isinstance(init_kwargs, dict):
        raise ValueError("--embeddings-init-kwargs-json must decode to a JSON object.")
    return cls(**init_kwargs)


def evaluate_dataset(
    dataset_jsonl: Path,
    answers_jsonl: Optional[Path],
    anthropic_model: str,
    embeddings_module: str,
    embeddings_class: str,
    embeddings_init_kwargs_json: Optional[str],
) -> None:
    rows = load_jsonl(dataset_jsonl)
    if answers_jsonl:
        rows = merge_answers(rows, load_jsonl(answers_jsonl))

    df = pd.DataFrame(rows)
    required_cols = ["question", "answer", "ground_truth", "contexts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    ragas_ds = Dataset.from_pandas(df[required_cols], preserve_index=False)

    llm = ChatAnthropic(model=anthropic_model, temperature=0)
    embeddings = load_custom_embeddings(
        module_path=embeddings_module,
        class_name=embeddings_class,
        init_kwargs_json=embeddings_init_kwargs_json,
    )

    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\nRAGAS Result:")
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate RAGAS dataset from support email threads")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-dataset", help="Extract issue/solution/context and build dataset JSONL")
    gen.add_argument("--input-dir", type=Path, required=True)
    gen.add_argument("--output-jsonl", type=Path, required=True)
    gen.add_argument("--anthropic-model", type=str, default="claude-3-5-sonnet-latest")

    ev = sub.add_parser("evaluate", help="Run RAGAS on generated dataset")
    ev.add_argument("--dataset-jsonl", type=Path, required=True)
    ev.add_argument("--answers-jsonl", type=Path, default=None)
    ev.add_argument("--anthropic-model", type=str, default="claude-3-5-sonnet-latest")
    ev.add_argument("--embeddings-module", type=str, required=True, help="Python module path for your custom embeddings class")
    ev.add_argument("--embeddings-class", type=str, required=True, help="Class name of your custom embeddings implementation")
    ev.add_argument(
        "--embeddings-init-kwargs-json",
        type=str,
        default=None,
        help='Optional JSON object string for custom embeddings init kwargs, e.g. "{\"model\":\"foo\"}"',
    )

    args = parser.parse_args()

    if args.command == "generate-dataset":
        build_dataset(args.input_dir, args.output_jsonl, args.anthropic_model)
    elif args.command == "evaluate":
        evaluate_dataset(
            args.dataset_jsonl,
            args.answers_jsonl,
            args.anthropic_model,
            args.embeddings_module,
            args.embeddings_class,
            args.embeddings_init_kwargs_json,
        )


if __name__ == "__main__":
    main()
