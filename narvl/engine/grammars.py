"""
GBNF (GGML BNF) Grammar Definitions for Strict JSON Generation in NARVL.

Guarantees 100% compliant JSON outputs from local SLM with zero markdown,
strictly validated fields, and enforced action enums.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set

VALID_ACTIONS: Set[str] = {
    "standardize_values",
    "regex_replace",
    "knn_impute",
    "deduplicate_exact",
    "clamp_bounds",
}

REQUIRED_STEP_KEYS: Set[str] = {
    "step_id",
    "target_column",
    "issue",
    "rule",
    "action",
    "parameters",
    "confidence",
    "justification",
    "loss_potential",
    "test_criterion",
}

# Strict GBNF Grammar for llama-cpp-python
CLEANING_PLAN_GBNF = r'''
root ::= "[" ws (step ("," ws step)*)? ws "]"

step ::= "{" ws
  "\"step_id\":" ws integer "," ws
  "\"target_column\":" ws string "," ws
  "\"issue\":" ws string "," ws
  "\"rule\":" ws string "," ws
  "\"action\":" ws action "," ws
  "\"parameters\":" ws object "," ws
  "\"confidence\":" ws float "," ws
  "\"justification\":" ws string "," ws
  "\"loss_potential\":" ws string "," ws
  "\"test_criterion\":" ws string ws
"}"

action ::= "\"standardize_values\"" | "\"regex_replace\"" | "\"knn_impute\"" | "\"deduplicate_exact\"" | "\"clamp_bounds\""

object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= string ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
value ::= string | number | object | array | "true" | "false" | "null"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
integer ::= [0-9]+
float ::= "0." [0-9]+ | "1.0" | "0" | "1"
ws ::= [ \t\n\r]*
'''


class GrammarValidationError(ValueError):
    """Raised when an emitted plan fails the GBNF schema."""


def validate_cleaning_step(step: Dict[str, Any]) -> bool:
    """Validate that a cleaning step dictionary strictly matches the GBNF schema."""
    if not isinstance(step, dict):
        raise GrammarValidationError(f"Step must be a dict, got {type(step)}")

    missing = REQUIRED_STEP_KEYS - set(step.keys())
    if missing:
        raise GrammarValidationError(f"Step is missing required keys: {missing}")

    if not isinstance(step["step_id"], int):
        raise GrammarValidationError("step_id must be an integer")
    if not isinstance(step["target_column"], str):
        raise GrammarValidationError("target_column must be a string")
    if not isinstance(step["issue"], str):
        raise GrammarValidationError("issue must be a string")
    if not isinstance(step["rule"], str):
        raise GrammarValidationError("rule must be a string")
    if step["action"] not in VALID_ACTIONS:
        raise GrammarValidationError(f"Invalid action: {step['action']}. Must be one of {VALID_ACTIONS}")
    if not isinstance(step["parameters"], dict):
        raise GrammarValidationError("parameters must be a dictionary")
    if step["action"] == "standardize_values":
        mapping = step["parameters"].get("mapping")
        if not isinstance(mapping, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in mapping.items()
        ):
            raise GrammarValidationError(
                "standardize_values parameters.mapping must be a string-to-string dictionary"
            )
    if not isinstance(step["confidence"], (int, float)):
        raise GrammarValidationError("confidence must be a float")
    if not (0.0 <= float(step["confidence"]) <= 1.0):
        raise GrammarValidationError("confidence must be between 0.0 and 1.0")
    if not isinstance(step["justification"], str):
        raise GrammarValidationError("justification must be a string")
    if not isinstance(step["loss_potential"], str):
        raise GrammarValidationError("loss_potential must be a string")
    if not isinstance(step["test_criterion"], str):
        raise GrammarValidationError("test_criterion must be a string")

    return True


def validate_cleaning_plan(plan: List[Dict[str, Any]]) -> bool:
    """Validate a list of cleaning steps."""
    if not isinstance(plan, list):
        raise GrammarValidationError(f"Plan must be a list of steps, got {type(plan)}")
    for step in plan:
        validate_cleaning_step(step)
    return True
