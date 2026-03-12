import argparse
import asyncio
import importlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from anthropic import Anthropic, AsyncAnthropic

from ragas import EvaluationDataset, experiment
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
)

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
- If no clear solution exists, set solution_original/ground_truth/answer to "NOT_AVAILABLE".
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
                issue_original=str(extracted.get("issue_original", "")).strip(),
                question=str(extracted.get("question", "")).strip(),
                solution_original=str(extracted.get("solution_original", "")).strip(),
                ground_truth=str(extracted.get("ground_truth", "")).strip(),
                answer=str(extracted.get("answer", "")).strip(),
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
    by_source = {
        r["source_file"]: r["answer"]
        for r in answers_rows
        if "source_file" in r and "answer" in r
    }
    merged = []
    for row in dataset_rows:
        row = dict(row)
        if row.get("source_file") in by_source:
            row["answer"] = by_source[row["source_file"]]
        merged.append(row)
    return merged


def load_custom_embeddings(
    module_path: str,
    class_name: str,
    init_kwargs_json: Optional[str],
):
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    init_kwargs = json.loads(init_kwargs_json) if init_kwargs_json else {}
    if not isinstance(init_kwargs, dict):
        raise ValueError("--embeddings-init-kwargs-json must decode to a JSON object.")
    return cls(**init_kwargs)


def _normalize_contexts(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        return [text]

    return [str(value).strip()]


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

    asyncio.run(
        _evaluate_dataset_async(
            rows=rows,
            anthropic_model=anthropic_model,
            embeddings_module=embeddings_module,
            embeddings_class=embeddings_class,
            embeddings_init_kwargs_json=embeddings_init_kwargs_json,
        )
    )


async def _evaluate_dataset_async(
    rows: List[dict],
    anthropic_model: str,
    embeddings_module: str,
    embeddings_class: str,
    embeddings_init_kwargs_json: Optional[str],
) -> None:
    df = pd.DataFrame(rows)

    required_cols = ["question", "answer", "ground_truth", "contexts"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df[required_cols].copy()
    df["question"] = df["question"].fillna("").astype(str).str.strip()
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str).str.strip()
    df["contexts"] = df["contexts"].apply(_normalize_contexts)

    invalid_mask = (
        df["question"].eq("")
        | df["answer"].eq("")
        | df["ground_truth"].eq("")
        | df["contexts"].apply(len).eq(0)
    )

    if invalid_mask.any():
        dropped = int(invalid_mask.sum())
        print(f"Dropping {dropped} invalid rows before evaluation")
        df = df.loc[~invalid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows left after preprocessing")

    records = df.rename(
        columns={
            "question": "user_input",
            "answer": "response",
            "ground_truth": "reference",
            "contexts": "retrieved_contexts",
        }
    ).to_dict(orient="records")

    eval_dataset = EvaluationDataset.from_list(records)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    anthropic_client = AsyncAnthropic(api_key=api_key)
    llm = llm_factory(
        anthropic_model,
        provider="anthropic",
        client=anthropic_client,
    )

    embeddings = load_custom_embeddings(
        module_path=embeddings_module,
        class_name=embeddings_class,
        init_kwargs_json=embeddings_init_kwargs_json,
    )

    faithfulness_metric = Faithfulness(llm=llm)
    answer_relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    answer_correctness_metric = AnswerCorrectness(llm=llm, embeddings=embeddings)
    factual_correctness_metric = FactualCorrectness(llm=llm)
    context_precision_metric = ContextPrecision(llm=llm)
    context_recall_metric = ContextRecall(llm=llm)

    @experiment(name="rag_eval")
    async def rag_eval(row: dict) -> dict:
        faithfulness_res = await faithfulness_metric.ascore(
            user_input=row["user_input"],
            response=row["response"],
            retrieved_contexts=row["retrieved_contexts"],
        )

        answer_relevancy_res = await answer_relevancy_metric.ascore(
            user_input=row["user_input"],
            response=row["response"],
        )

        answer_correctness_res = await answer_correctness_metric.ascore(
            user_input=row["user_input"],
            response=row["response"],
            reference=row["reference"],
        )

        factual_correctness_res = await factual_correctness_metric.ascore(
            response=row["response"],
            reference=row["reference"],
        )

        context_precision_res = await context_precision_metric.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )

        context_recall_res = await context_recall_metric.ascore(
            user_input=row["user_input"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )

        return {
            **row,
            "faithfulness": faithfulness_res.value,
            "faithfulness_reason": getattr(faithfulness_res, "reason", None),
            "answer_relevancy": answer_relevancy_res.value,
            "answer_relevancy_reason": getattr(answer_relevancy_res, "reason", None),
            "answer_correctness": answer_correctness_res.value,
            "answer_correctness_reason": getattr(answer_correctness_res, "reason", None),
            "factual_correctness": factual_correctness_res.value,
            "factual_correctness_reason": getattr(factual_correctness_res, "reason", None),
            "context_precision": context_precision_res.value,
            "context_precision_reason": getattr(context_precision_res, "reason", None),
            "context_recall": context_recall_res.value,
            "context_recall_reason": getattr(context_recall_res, "reason", None),
        }

    results = await rag_eval.arun(eval_dataset)
    result_df = results.to_pandas()

    print("\nPer-row RAGAS results:")
    print(result_df)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "factual_correctness",
        "context_precision",
        "context_recall",
    ]

    summary = {
        col: float(result_df[col].dropna().mean())
        for col in metric_cols
        if col in result_df.columns
    }

    print("\nMean metric scores:")
    print(summary)

    output_csv = Path("ragas_eval_results.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved per-row results to {output_csv.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate RAGAS dataset from support threads"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate-dataset",
        help="Extract issue/solution/context and build dataset JSONL",
    )
    gen.add_argument("--input-dir", type=Path, required=True)
    gen.add_argument("--output-jsonl", type=Path, required=True)
    gen.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-3-5-sonnet-latest",
    )

    ev = sub.add_parser("evaluate", help="Run RAGAS on generated dataset")
    ev.add_argument("--dataset-jsonl", type=Path, required=True)
    ev.add_argument("--answers-jsonl", type=Path, default=None)
    ev.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-3-5-sonnet-latest",
    )
    ev.add_argument(
        "--embeddings-module",
        type=str,
        required=True,
        help="Python module path for your embeddings class",
    )
    ev.add_argument(
        "--embeddings-class",
        type=str,
        required=True,
        help="Class name of your embeddings implementation",
    )
    ev.add_argument(
        "--embeddings-init-kwargs-json",
        type=str,
        default=None,
        help='Optional JSON object string for custom embeddings init kwargs, e.g. \'{"model":"text-embedding-3-large"}\'',
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
