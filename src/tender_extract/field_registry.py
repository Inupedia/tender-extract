"""统一字段注册表，避免 Router / Rule / LLM 三套字段定义漂移。"""
from __future__ import annotations

from dataclasses import dataclass

from .module_router import TENDER_MODULES
from .patterns import FIELD_PATTERNS


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    modules: tuple[str, ...] = ()
    has_rule_patterns: bool = False
    llm_recoverable: bool = True


def _build_registry() -> dict[str, FieldDefinition]:
    module_map: dict[str, set[str]] = {}
    for module in TENDER_MODULES:
        for field_name in module.target_fields:
            module_map.setdefault(field_name, set()).add(module.module_id)

    names = set(FIELD_PATTERNS) | set(module_map)
    return {
        name: FieldDefinition(
            name=name,
            modules=tuple(sorted(module_map.get(name, set()))),
            has_rule_patterns=name in FIELD_PATTERNS,
            llm_recoverable=name != "table_data",
        )
        for name in sorted(names)
    }


FIELD_REGISTRY = _build_registry()


def get_rule_fields() -> set[str]:
    return {name for name, definition in FIELD_REGISTRY.items() if definition.has_rule_patterns}


def get_expected_fields_for_modules(module_ids: set[str]) -> set[str]:
    if not module_ids:
        return set()
    return {
        name
        for name, definition in FIELD_REGISTRY.items()
        if definition.llm_recoverable and module_ids.intersection(definition.modules)
    }
