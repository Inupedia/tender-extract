"""Gold-dataset evaluation for extraction quality.

The evaluator deliberately scores only fields explicitly annotated in the gold
case. This makes partially-labelled datasets safe to grow over time without
penalising unrelated extracted fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .pipeline import ExtractionPipeline
from .schema import ExtractionResult, ProcessingConfig


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    document: Path
    expected: dict[str, list[str]]
    tags: tuple[str, ...] = ()


@dataclass
class FieldMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def add(self, other: "FieldMetrics") -> None:
        self.true_positive += other.true_positive
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class EvaluationFailure:
    case_id: str
    field_name: str
    expected: tuple[str, ...]
    predicted: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "field_name": self.field_name,
            "expected": list(self.expected),
            "predicted": list(self.predicted),
        }


@dataclass
class EvaluationReport:
    cases: int
    exact_cases: int
    micro: FieldMetrics
    per_field: dict[str, FieldMetrics]
    failures: list[EvaluationFailure] = field(default_factory=list)
    provider: str = "none"
    model: str | None = None
    llm_calls: int = 0

    @property
    def exact_case_accuracy(self) -> float:
        return self.exact_cases / self.cases if self.cases else 0.0

    @property
    def macro_f1(self) -> float:
        if not self.per_field:
            return 0.0
        return sum(metrics.f1 for metrics in self.per_field.values()) / len(self.per_field)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "exact_cases": self.exact_cases,
            "exact_case_accuracy": self.exact_case_accuracy,
            "micro": self.micro.as_dict(),
            "macro_f1": self.macro_f1,
            "per_field": {
                name: metrics.as_dict()
                for name, metrics in sorted(self.per_field.items())
            },
            "failures": [failure.as_dict() for failure in self.failures],
            "llm": {
                "provider": self.provider,
                "model": self.model,
                "calls": self.llm_calls,
            },
        }


def normalize_value(value: str) -> str:
    """Normalise superficial formatting while keeping semantic differences visible."""
    value = str(value or "").strip().casefold()
    value = value.replace("（", "(").replace("）", ")")
    value = value.replace("，", ",").replace("：", ":")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[\u200b\ufeff]", "", value)
    return value


def load_gold_dataset(dataset_path: str | Path) -> list[GoldCase]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Gold dataset does not exist: {path}")

    cases: list[GoldCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

        case_id = str(payload.get("id") or f"case-{line_number}")
        document_raw = payload.get("document")
        expected_raw = payload.get("expected")
        if not document_raw or not isinstance(expected_raw, dict):
            raise ValueError(
                f"Gold case {case_id} must contain 'document' and object 'expected'"
            )

        expected: dict[str, list[str]] = {}
        for field_name, values in expected_raw.items():
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Gold case {case_id}/{field_name} must be string or list[str]")
            expected[str(field_name)] = [value for value in values if value.strip()]

        document = Path(str(document_raw))
        if not document.is_absolute():
            document = path.parent / document
        cases.append(
            GoldCase(
                case_id=case_id,
                document=document,
                expected=expected,
                tags=tuple(str(tag) for tag in (payload.get("tags") or [])),
            )
        )
    if not cases:
        raise ValueError(f"Gold dataset is empty: {path}")
    return cases


def _prediction_values(result: ExtractionResult, field_name: str) -> list[str]:
    extracted = result.fields.get(field_name)
    if not extracted:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for span in extracted.values:
        value = span.normalized_value or span.value
        normalized = normalize_value(value)
        if normalized and normalized not in seen:
            values.append(value)
            seen.add(normalized)
    if not values and extracted.primary_value:
        values.append(extracted.primary_value)
    return values


def _score_values(expected: Iterable[str], predicted: Iterable[str]) -> FieldMetrics:
    expected_values = {normalize_value(value) for value in expected if normalize_value(value)}
    predicted_values = {normalize_value(value) for value in predicted if normalize_value(value)}
    matched = expected_values.intersection(predicted_values)
    return FieldMetrics(
        true_positive=len(matched),
        false_positive=len(predicted_values - matched),
        false_negative=len(expected_values - matched),
    )


def score_case(
    case: GoldCase,
    result: ExtractionResult,
) -> tuple[dict[str, FieldMetrics], list[EvaluationFailure], bool]:
    per_field: dict[str, FieldMetrics] = {}
    failures: list[EvaluationFailure] = []
    exact = True

    for field_name, expected in case.expected.items():
        predicted = _prediction_values(result, field_name)
        metrics = _score_values(expected, predicted)
        per_field[field_name] = metrics
        if metrics.false_positive or metrics.false_negative:
            exact = False
            failures.append(
                EvaluationFailure(
                    case_id=case.case_id,
                    field_name=field_name,
                    expected=tuple(expected),
                    predicted=tuple(predicted),
                )
            )
    return per_field, failures, exact


def evaluate_dataset(
    dataset_path: str | Path,
    config: ProcessingConfig,
) -> EvaluationReport:
    cases = load_gold_dataset(dataset_path)
    pipeline = ExtractionPipeline(config)
    aggregate = FieldMetrics()
    per_field: dict[str, FieldMetrics] = {}
    failures: list[EvaluationFailure] = []
    exact_cases = 0
    llm_calls = 0

    for case in cases:
        if not case.document.exists():
            raise FileNotFoundError(
                f"Gold case {case.case_id} document does not exist: {case.document}"
            )
        result = pipeline.extract_file(str(case.document))
        llm_calls += result.llm_calls
        case_metrics, case_failures, exact = score_case(case, result)
        if exact:
            exact_cases += 1
        failures.extend(case_failures)
        for field_name, metrics in case_metrics.items():
            aggregate.add(metrics)
            per_field.setdefault(field_name, FieldMetrics()).add(metrics)

    return EvaluationReport(
        cases=len(cases),
        exact_cases=exact_cases,
        micro=aggregate,
        per_field=per_field,
        failures=failures,
        provider=config.llm_provider,
        model=config.llm_model,
        llm_calls=llm_calls,
    )
