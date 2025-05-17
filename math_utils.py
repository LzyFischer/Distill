# -*- coding: utf-8 -*-
"""Robust math-equivalence utilities
====================================
Handles:
• trailing *units* (plain and LaTeX `\\text{…}`)
• *percentages* → decimal conversion (``12%`` → ``0.12``)
• *fractions* (inline ``-3/5`` or ``\\frac{…}{…}``)
• negatives, radicals, π, degrees, boxed answers, etc.
Public API preserved: ``floatify``, ``within_eps``, ``parse_boxed``,
``parse_math_boxed``, ``is_math_correct``, ``evaluate_math``.
"""
from __future__ import annotations
import re, signal
from typing import Any, Optional
from latex2sympy2 import latex2sympy
from sympy import simplify, sympify
import pdb

# ───────────────────────────── REGEXES ────────────────────────────────
_NUM_TOKEN   = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")
_PLAIN_UNIT  = re.compile(r"(?P<num>[-+]?\d*\.?\d+)\s*(?P<unit>[a-zA-Z°]+(?:\^\d+)?|%)")
_TEXT_UNIT   = re.compile(r"\\text\s*{[^}]*}")
_PERCENT_LATEX = re.compile(r"([-+]?\d*\.?\d+)\\%")  # e.g. 12\%
_PERCENT_PLAIN = re.compile(r"([-+]?\d*\.?\d+)%")       # e.g. 12%

# --------------------------------------------------------------------
# BASIC HELPERS
# --------------------------------------------------------------------

def _convert_percentage(expr: str) -> str:
    """Replace percentage tokens with decimal equivalent."""
    expr = _PERCENT_LATEX.sub(lambda m: str(float(m.group(1)) / 100), expr)
    expr = _PERCENT_PLAIN.sub(lambda m: str(float(m.group(1)) / 100), expr)
    return expr


def _strip_units(expr: str | None) -> str:
    if not expr:
        return ""
    expr = _TEXT_UNIT.sub("", expr)
    def _unit_replacer(m):
        return m.group("num")
    return _PLAIN_UNIT.sub(_unit_replacer, expr)


def floatify(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def within_eps(pred: str | float, gt: str | float, eps: float = 1e-2) -> bool:
    p, g = floatify(pred), floatify(gt)
    return p is not None and g is not None and abs(p - g) < eps

# --------------------------------------------------------------------
# STRING NORMALISATION & PARSERS
# --------------------------------------------------------------------
_RE_BRACE_CONTENT = re.compile(r"^\\(?:boxed|fbox){(.+)}$")


def _last_boxed(expr: str) -> str | None:
    if not expr:
        return None
    idx = max(expr.rfind("\\boxed"), expr.rfind("\\fbox"))
    if idx == -1:
        return None
    depth = 0
    for i in range(idx, len(expr)):
        if expr[i] == "{":
            depth += 1
        elif expr[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = expr[idx:i + 1]
                m = _RE_BRACE_CONTENT.match(candidate)
                if m:
                    return m.group(1)
                break
    return None


def _normalise(expr: str | None) -> str:
    if not expr:
        return ""
    try:
        expr = _convert_percentage(expr)
    except Exception:
        pass
    expr = expr.lower()
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("^{\\circ}", "").replace("^\\circ", "")
    expr = expr.replace("\\$", "")
    expr = expr.replace(" ", "")
    expr = expr.replace("tfrac", "frac").replace("dfrac", "frac")
    expr = _strip_units(expr)
    # simple X/Y → \frac{X}{Y}
    if "/" in expr and "\\frac" not in expr:
        parts = expr.split("/")
        if len(parts) == 2 and all(p.strip("-+").isdigit() for p in parts):
            expr = f"\\frac{{{parts[0]}}}{{{parts[1]}}}"
    return expr


def _numeric_val(expr: str) -> Optional[float]:
    try:
        return float(latex2sympy(expr).evalf())
    except Exception:
        tokens = _NUM_TOKEN.findall(expr)
        return floatify(tokens[-1]) if tokens else None

# --------------------------------------------------------------------
# PUBLIC PARSE HELPERS (kept for backward-compat)
# --------------------------------------------------------------------

def parse_math_boxed(s: str | None):
    """Return *raw* content inside the last \boxed/\fbox; else "N/A"."""
    return _last_boxed(s) or "N/A"


def parse_boxed(s: str | None):
    """Return numeric value of last boxed content, or "N/A"."""
    content = _last_boxed(s)
    if content is None:
        return "N/A"
    val = _numeric_val(_normalise(content))
    return val if val is not None else "N/A"

# --------------------------------------------------------------------
# MAIN COMPARATOR
# --------------------------------------------------------------------

def _vals_equiv(a: str, b: str) -> bool:
    va, vb = _numeric_val(a), _numeric_val(b)
    return va is not None and vb is not None and within_eps(va, vb)


def is_math_correct(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    n_pred, n_gt = _normalise(pred), _normalise(gt)
    # 1) identical strings after normalisation
    if n_pred == n_gt:
        return True
    # 2) numeric equivalence
    if _vals_equiv(n_pred, n_gt):
        return True
    # 3) compare boxed content to other expression
    p_box, g_box = _last_boxed(pred), _last_boxed(gt)
    if p_box and _vals_equiv(_normalise(p_box), n_gt):
        return True
    if g_box and _vals_equiv(n_pred, _normalise(g_box)):
        return True
    if p_box and g_box and _vals_equiv(_normalise(p_box), _normalise(g_box)):
        return 
    def _strip_pct(s: str) -> str:
        return s.replace(r"\%", "").replace("%", "")
    n_pred_no_pct = _normalise(_strip_pct(pred))
    n_gt_no_pct   = _normalise(_strip_pct(gt))
    if n_pred_no_pct == n_gt_no_pct or _vals_equiv(n_pred_no_pct, n_gt_no_pct):
        return True
    return False

# --------------------------------------------------------------------
# TIMEOUT-PROTECTED WRAPPER
# --------------------------------------------------------------------
_TIMEOUT = 5

def _alarm_handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, _alarm_handler)

def safe_is_math_correct(pred: str, gt: str) -> bool:
    signal.alarm(_TIMEOUT)
    try:
        return is_math_correct(pred, gt)
    finally:
        signal.alarm(0)

# --------------------------------------------------------------------
# BULK ACCURACY
# --------------------------------------------------------------------

def evaluate_math(results: list[dict[str, str]]) -> float:
    correct = sum(safe_is_math_correct(r["pred"], r["gold_answer"]) for r in results)
    return round(correct / max(len(results), 1), 4)