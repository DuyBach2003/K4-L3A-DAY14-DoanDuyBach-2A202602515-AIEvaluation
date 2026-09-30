"""Exercise 3.4 (bonus) — RAGAS vs DeepEval on the same saved RAG traces.

Both frameworks score the identical inputs from ``golden_dataset.json`` and
``artifacts/actual_answers.json`` with the same judge model, so differences in
scores come from the frameworks' metric definitions, not from the data.

This script is optional bonus work. It is not imported by ``template.py`` or the
tests, and its dependencies live in ``bonus/requirements-bonus.txt`` instead of
the lab's ``requirements.txt``.

Usage (from the repo root, in an environment with the bonus requirements):

    python bonus/compare_frameworks.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Awaitable, Callable

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
# Closest heuristic counterpart in template.py for each framework metric.
HEURISTIC_FIELDS: dict[str, str] = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "relevance",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
}
FAIL_THRESHOLD = 0.5
JUDGE_MAX_TOKENS = 2048


@dataclass
class Trace:
    """One golden record joined with the assistant's saved answer."""

    id: str
    question: str
    expected: str
    answer: str
    contexts: list[str]


def load_traces(golden_path: Path, actual_path: Path) -> list[Trace]:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    actual_by_id = {record["id"]: record for record in actual["answers"]}
    traces: list[Trace] = []
    for record in golden["qa_pairs"]:
        saved = actual_by_id[record["id"]]
        if saved["question"] != record["question"]:
            raise ValueError(f"{record['id']}: question differs between artifacts")
        traces.append(
            Trace(
                id=record["id"],
                question=record["question"],
                expected=record["expected_answer"],
                answer=saved["actual_answer"],
                contexts=[c["text"] for c in saved["retrieved_contexts"]],
            )
        )
    return traces


async def _guarded(
    semaphore: asyncio.Semaphore,
    call: Callable[[], Awaitable[float]],
) -> tuple[float | None, str | None]:
    """Run one metric call; a framework error becomes (None, message).

    Scores are rounded so RAGAS's 0.9999999999 compares equal to DeepEval's 1.0.
    """
    async with semaphore:
        try:
            return round(float(await call()), 4), None
        except Exception as exc:  # noqa: BLE001 - record and keep the batch going
            return None, f"{type(exc).__name__}: {exc}"


async def run_ragas(
    traces: list[Trace],
    model: str,
    embedding_model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, dict[str, Any]]:
    client = AsyncOpenAI()
    llm = llm_factory(model, client=client, temperature=0)
    embeddings = OpenAIEmbeddings(client=client, model=embedding_model)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    recall = ContextRecall(llm=llm)
    precision = ContextPrecisionWithReference(llm=llm)

    async def value(coro: Awaitable[Any]) -> float:
        return (await coro).value

    async def score(trace: Trace) -> dict[str, Any]:
        calls: dict[str, Callable[[], Awaitable[float]]] = {
            "faithfulness": lambda: value(
                faithfulness.ascore(
                    user_input=trace.question,
                    response=trace.answer,
                    retrieved_contexts=trace.contexts,
                )
            ),
            "answer_relevancy": lambda: value(
                relevancy.ascore(user_input=trace.question, response=trace.answer)
            ),
            "context_recall": lambda: value(
                recall.ascore(
                    user_input=trace.question,
                    retrieved_contexts=trace.contexts,
                    reference=trace.expected,
                )
            ),
            "context_precision": lambda: value(
                precision.ascore(
                    user_input=trace.question,
                    reference=trace.expected,
                    retrieved_contexts=trace.contexts,
                )
            ),
        }
        outcomes = await asyncio.gather(
            *(_guarded(semaphore, calls[name]) for name in METRICS)
        )
        row: dict[str, Any] = {"errors": {}}
        for name, (result, error) in zip(METRICS, outcomes):
            row[name] = result
            if error:
                row["errors"][name] = error
        return row

    rows = await asyncio.gather(*(score(trace) for trace in traces))
    return {trace.id: row for trace, row in zip(traces, rows)}


class JsonModeJudge(DeepEvalBaseLLM):
    """DeepEval judge that uses OpenAI JSON mode with a bounded output length.

    DeepEval's built-in ``GPTModel`` sends its pydantic schemas as strict
    structured output. With gpt-4o-mini the faithfulness ``Verdicts`` schema
    (optional ``reason`` field) makes the model emit whitespace until the token
    limit, and ``GPTModel`` retries ``LengthFinishReasonError`` without a stop
    condition. JSON mode plus local validation avoids that loop; a truncated
    response fails once and is recorded as an error for that metric.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._client = OpenAI(max_retries=2, timeout=60)
        self._async_client = AsyncOpenAI(max_retries=2, timeout=60)
        super().__init__(model)

    def load_model(self) -> AsyncOpenAI:
        return self._async_client

    def _request(self, prompt: str, schema: type[BaseModel] | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": JUDGE_MAX_TOKENS,
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}
        return request

    @staticmethod
    def _parse(completion: Any, schema: type[BaseModel] | None) -> Any:
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("judge response was truncated at the token limit")
        content = choice.message.content or ""
        return content if schema is None else schema.model_validate_json(content)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        completion = self._client.chat.completions.create(
            **self._request(prompt, schema)
        )
        return self._parse(completion, schema)

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> Any:
        completion = await self._async_client.chat.completions.create(
            **self._request(prompt, schema)
        )
        return self._parse(completion, schema)

    def get_model_name(self) -> str:
        return f"{self._model} (JSON mode)"


async def run_deepeval(
    traces: list[Trace],
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, dict[str, Any]]:
    judge = JsonModeJudge(model)
    metric_classes: dict[str, type] = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "context_recall": ContextualRecallMetric,
        "context_precision": ContextualPrecisionMetric,
    }

    async def score(trace: Trace) -> dict[str, Any]:
        test_case = LLMTestCase(
            input=trace.question,
            actual_output=trace.answer,
            expected_output=trace.expected,
            retrieval_context=trace.contexts,
        )
        # DeepEval metrics keep per-measurement state, so use one instance per call.
        metrics = {
            name: metric_class(model=judge, threshold=FAIL_THRESHOLD)
            for name, metric_class in metric_classes.items()
        }

        def make_call(name: str) -> Callable[[], Awaitable[float]]:
            async def call() -> float:
                await metrics[name].a_measure(test_case, _show_indicator=False)
                return metrics[name].score

            return call

        outcomes = await asyncio.gather(
            *(_guarded(semaphore, make_call(name)) for name in METRICS)
        )
        row: dict[str, Any] = {"errors": {}, "reasons": {}}
        for name, (result, error) in zip(METRICS, outcomes):
            row[name] = result
            if error:
                row["errors"][name] = error
            else:
                row["reasons"][name] = metrics[name].reason
        return row

    rows = await asyncio.gather(*(score(trace) for trace in traces))
    return {trace.id: row for trace, row in zip(traces, rows)}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = mean(xs), mean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / (var_x * var_y) ** 0.5


def summarize(
    ids: list[str],
    ragas: dict[str, dict[str, Any]],
    deepeval: dict[str, dict[str, Any]],
    heuristic: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Per-metric agreement between the two frameworks plus failure overlap."""
    summary: dict[str, Any] = {"metrics": {}, "fail_threshold": FAIL_THRESHOLD}
    for name in METRICS:
        paired = [
            (case_id, ragas[case_id][name], deepeval[case_id][name])
            for case_id in ids
            if ragas[case_id][name] is not None and deepeval[case_id][name] is not None
        ]
        ragas_scores = [r for _, r, _ in paired]
        deepeval_scores = [d for _, _, d in paired]
        ragas_fail = sorted(i for i, r, _ in paired if r < FAIL_THRESHOLD)
        deepeval_fail = sorted(i for i, _, d in paired if d < FAIL_THRESHOLD)
        heuristic_scores = [
            heuristic[i][HEURISTIC_FIELDS[name]]
            for i, _, _ in paired
            if i in heuristic and heuristic[i].get(HEURISTIC_FIELDS[name]) is not None
        ]
        summary["metrics"][name] = {
            "n": len(paired),
            "avg_ragas": mean(ragas_scores) if paired else None,
            "avg_deepeval": mean(deepeval_scores) if paired else None,
            "avg_heuristic": mean(heuristic_scores) if heuristic_scores else None,
            "mean_abs_diff": (
                mean(abs(r - d) for r, d in zip(ragas_scores, deepeval_scores))
                if paired
                else None
            ),
            "pearson": _pearson(ragas_scores, deepeval_scores),
            "ragas_stricter_cases": sum(r < d for r, d in zip(ragas_scores, deepeval_scores)),
            "deepeval_stricter_cases": sum(d < r for r, d in zip(ragas_scores, deepeval_scores)),
            "ragas_fail_ids": ragas_fail,
            "deepeval_fail_ids": deepeval_fail,
            "both_fail_ids": sorted(set(ragas_fail) & set(deepeval_fail)),
        }

    def failing_cases(scores: dict[str, dict[str, Any]]) -> list[str]:
        return sorted(
            case_id
            for case_id in ids
            if any(
                scores[case_id][name] is not None
                and scores[case_id][name] < FAIL_THRESHOLD
                for name in METRICS
            )
        )

    ragas_failing = failing_cases(ragas)
    deepeval_failing = failing_cases(deepeval)
    summary["any_metric_fail"] = {
        "ragas": ragas_failing,
        "deepeval": deepeval_failing,
        "both": sorted(set(ragas_failing) & set(deepeval_failing)),
        "ragas_only": sorted(set(ragas_failing) - set(deepeval_failing)),
        "deepeval_only": sorted(set(deepeval_failing) - set(ragas_failing)),
        "heuristic": sorted(i for i in ids if i in heuristic and not heuristic[i]["passed"]),
    }
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_report(
    ids: list[str],
    ragas: dict[str, dict[str, Any]],
    deepeval: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    print("| ID | Faith R | Faith D | AnsRel R | AnsRel D | Recall R | Recall D | Prec R | Prec D |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case_id in ids:
        cells = " | ".join(
            f"{_fmt(ragas[case_id][name])} | {_fmt(deepeval[case_id][name])}"
            for name in METRICS
        )
        print(f"| {case_id} | {cells} |")

    print("\n| Metric | n | Avg RAGAS | Avg DeepEval | Avg heuristic | Mean abs diff | Pearson r | RAGAS fails | DeepEval fails |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for name in METRICS:
        row = summary["metrics"][name]
        print(
            f"| {name} | {row['n']} | {_fmt(row['avg_ragas'])} | "
            f"{_fmt(row['avg_deepeval'])} | {_fmt(row['avg_heuristic'])} | "
            f"{_fmt(row['mean_abs_diff'])} | {_fmt(row['pearson'])} | "
            f"{', '.join(row['ragas_fail_ids']) or '-'} | "
            f"{', '.join(row['deepeval_fail_ids']) or '-'} |"
        )
    print(f"\nAny metric < {FAIL_THRESHOLD}: {json.dumps(summary['any_metric_fail'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RAGAS and DeepEval on the saved lab traces."
    )
    parser.add_argument("--golden", type=Path, default=REPO_ROOT / "golden_dataset.json")
    parser.add_argument(
        "--actual", type=Path, default=REPO_ROOT / "artifacts" / "actual_answers.json"
    )
    parser.add_argument(
        "--heuristic",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_results.json",
        help="Exercise 3.2 results, used only as a reference column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "framework_comparison.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N cases")
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing from .env")
        return 2
    model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    embedding_model = "text-embedding-3-small"

    traces = load_traces(args.golden, args.actual)[: args.limit]
    ids = [trace.id for trace in traces]
    heuristic: dict[str, dict[str, Any]] = {}
    if args.heuristic.exists():
        saved = json.loads(args.heuristic.read_text(encoding="utf-8"))
        heuristic = {row["id"]: row for row in saved["results"]}

    semaphore = asyncio.Semaphore(args.concurrency)
    ragas = await run_ragas(traces, model, embedding_model, semaphore)
    deepeval = await run_deepeval(traces, model, semaphore)
    summary = summarize(ids, ragas, deepeval, heuristic)

    import deepeval as deepeval_package
    import ragas as ragas_package

    artifact = {
        "judge_model": model,
        "embedding_model": embedding_model,
        "versions": {
            "ragas": ragas_package.__version__,
            "deepeval": deepeval_package.__version__,
        },
        "cases": [
            {"id": case_id, "ragas": ragas[case_id], "deepeval": deepeval[case_id]}
            for case_id in ids
        ],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_report(ids, ragas, deepeval, summary)
    print(f"\nSaved framework comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
