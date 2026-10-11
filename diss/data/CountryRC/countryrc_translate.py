"""Translate CountryRC items into ko/es/zh/ar/ja/fr via OpenAI, cached per locale."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Formatter, Template

BASE = Path(__file__).resolve().parent

DEFAULT_INPUT = BASE / "data.json"
DEFAULT_OUTPUT = BASE / "data_multilingual.json"
DEFAULT_CACHE_DIR = BASE / "translation_cache_gpt"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.5"
OPENAI_REASONING_EFFORT = "low"
OPENAI_TIMEOUT = 180
LLM_RETRIES = 2

ENGLISH_LOCALE = "en"

LOCALES = {
    "ko": dict(lang="Korean"),
    "es": dict(lang="Spanish"),
    "zh": dict(lang="Chinese (Simplified)"),
    "ar": dict(lang="Arabic"),
    "ja": dict(lang="Japanese"),
    "fr": dict(lang="French"),
}

LOCALE_HINTS = {
    "ko": "Korean: prefer `{country}에서` and other locative positions. Avoid particles whose form depends on 받침 (은/는, 이/가, 을/를, 와/과, 로/으로) directly after placeholders.",
    "ja": "Japanese: `{country}で` and `{country}に` are safe positions.",
    "ar": "Arabic: keep the placeholder as a standalone noun — no attached ال-, prefixes, case endings, or conjunctions.",
    "es": "Spanish: avoid gendered articles directly before placeholders. `en {country}` is usually safe.",
    "fr": "French: avoid `en/au/aux/à/de/du/des/d'` directly before placeholders — these vary by country. Predicate forms like `le pays était {country}` are safe.",
    "zh": "Chinese: country names take no classifier; most positions work.",
}


def load_prompt(name: str, **kwargs) -> str:
    return Template((BASE / f"{name}.txt").read_text(encoding="utf-8")).substitute(
        **kwargs
    )


_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "context": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["context", "question"],
    "additionalProperties": False,
}


_usage = {"prompt": 0, "completion": 0, "calls": 0}
_usage_lock = threading.Lock()
_API_KEY: str | None = None


def _openai_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        f = Path.home() / ".openai_api_key"
        key = (f.read_text().strip() if f.exists() else "") or os.environ.get(
            "OPENAI_API_KEY", ""
        ).strip()
        if not key:
            raise RuntimeError(
                "no OpenAI API key in ~/.openai_api_key or $OPENAI_API_KEY"
            )
        _API_KEY = key
    return _API_KEY


def _call_openai_json(prompt: str, schema: dict) -> dict:
    body = json.dumps(
        {
            "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "max_completion_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "countryrc_translation",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
    ).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(1 + LLM_RETRIES):
        wait = 2 * (attempt + 1)
        try:
            req = urllib.request.Request(
                OPENAI_API_URL,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {_openai_key()}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=OPENAI_TIMEOUT) as resp:
                payload = json.loads(resp.read())
            choice = payload["choices"][0]
            msg = choice["message"]
            if msg.get("refusal"):
                raise RuntimeError(f"refusal: {msg['refusal']}")
            content = msg.get("content")
            if not content:
                raise RuntimeError(
                    f"empty content (finish={choice.get('finish_reason')})"
                )
            data = json.loads(content)
            usage = payload.get("usage") or {}
            with _usage_lock:
                _usage["prompt"] += usage.get("prompt_tokens", 0)
                _usage["completion"] += usage.get("completion_tokens", 0)
                _usage["calls"] += 1
            return data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code == 429:
                wait = 6 * (attempt + 1)
        except Exception as exc:
            last_exc = exc
        if attempt < LLM_RETRIES:
            print(
                f"[llm] call failed (attempt {attempt + 1}/"
                f"{1 + LLM_RETRIES}), retrying: {str(last_exc)[:160]}"
            )
            time.sleep(wait)
    raise RuntimeError(f"openai failed after {1 + LLM_RETRIES} attempts: {last_exc}")


def llm_translate(locale: str, context_en: str, question_en: str) -> dict[str, str]:
    prompt = load_prompt(
        "prompt_countryrc_translation",
        language_name=LOCALES[locale]["lang"],
        context_en=context_en,
        question_en=question_en,
        locale_hint=LOCALE_HINTS[locale],
    )
    data = _call_openai_json(prompt, _LLM_SCHEMA)
    return {"context": data["context"], "question": data["question"]}


def _validate(item: dict, translation: dict[str, str]) -> list[str]:
    """Placeholder-preservation check; returns issue strings (empty = OK)."""
    issues: list[str] = []
    if not isinstance(translation, dict):
        return ["translation must contain context and question"]
    for field in ("context", "question"):
        text = translation.get(field)
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{field} must be a nonempty string")
            continue
        try:
            original = Counter(
                (name, spec, conversion)
                for _, name, spec, conversion in Formatter().parse(item[field])
                if name is not None
            )
            translated = Counter(
                (name, spec, conversion)
                for _, name, spec, conversion in Formatter().parse(text)
                if name is not None
            )
        except ValueError as exc:
            issues.append(f"{field}: {exc}")
            continue
        if original != translated:
            issues.append(f"{field} placeholder counts do not match source")
        if any(name not in {"country", "country_dummy"} or spec or conversion
               for name, spec, conversion in translated):
            issues.append(f"{field} contains unsupported format fields")
    return issues


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def run(
    items: list[dict], locales: list[str], cache_dir: Path, workers: int
) -> dict[str, dict[str, dict[str, str]]]:
    caches: dict[str, dict[str, dict[str, str]]] = {}
    for loc in locales:
        caches[loc] = _load_cache(cache_dir / f"{loc}.json")

    jobs: list[tuple[str, str, dict]] = []
    for item in items:
        for loc in locales:
            if item["id"] in caches[loc]:
                issues = _validate(item, caches[loc][item["id"]])
                if issues:
                    raise ValueError(f"{loc}/{item['id']}: {'; '.join(issues)}")
            else:
                jobs.append((loc, item["id"], item))
    print(
        f"[plan] {len(items)} items × {len(locales)} locales = "
        f"{len(items) * len(locales)} pairs | "
        f"{len(jobs)} new jobs | "
        f"{len(items) * len(locales) - len(jobs)} cached"
    )

    if not jobs:
        return caches

    def _do(locale: str, item_id: str, item: dict):
        tr = llm_translate(locale, item["context"], item["question"])
        issues = _validate(item, tr)
        if issues:
            raise ValueError("; ".join(issues))
        return locale, item_id, tr

    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_do, *job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                locale, item_id, tr = fut.result()
            except Exception as exc:
                print(f"[llm] {job[0]}/{job[1]}: {exc}")
                failures.append(f"{job[0]}/{job[1]}: {exc}")
                continue
            caches[locale][item_id] = tr
            done += 1
            if done % 25 == 0:
                for loc in locales:
                    _save_cache(cache_dir / f"{loc}.json", caches[loc])
                print(f"[llm] {done}/{len(jobs)}")

    for loc in locales:
        _save_cache(cache_dir / f"{loc}.json", caches[loc])

    if failures:
        raise ValueError("Incomplete CountryRC translations: " + "; ".join(failures))
    print("[ok] all translations passed placeholder validation")
    return caches


def assemble(
    items: list[dict], caches: dict[str, dict[str, dict[str, str]]]
) -> list[dict]:
    out = []
    for item in items:
        record = {
            "id": item["id"],
            "translations": {
                ENGLISH_LOCALE: {
                    "context": item["context"],
                    "question": item["question"],
                },
            },
        }
        for loc, cache in caches.items():
            if item["id"] not in cache:
                raise ValueError(f"Missing CountryRC translation: {loc}/{item['id']}")
            issues = _validate(item, cache[item["id"]])
            if issues:
                raise ValueError(f"{loc}/{item['id']}: {'; '.join(issues)}")
            record["translations"][loc] = cache[item["id"]]
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--locales",
        nargs="*",
        default=None,
        help=f"Subset of {list(LOCALES)} (default: all)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N items (smoke runs)",
    )
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.locales is not None and not args.locales:
        p.error("--locales requires at least one locale")
    locales = args.locales if args.locales is not None else list(LOCALES)
    unknown = set(locales) - set(LOCALES)
    if unknown:
        p.error(f"unknown locales: {', '.join(sorted(unknown))}")
    canonical_output = args.output.resolve() == DEFAULT_OUTPUT.resolve() or (
        args.output.exists() and DEFAULT_OUTPUT.exists()
        and args.output.samefile(DEFAULT_OUTPUT)
    )
    if (args.limit is not None or args.locales is not None) and canonical_output:
        p.error("--limit/--locales requires --output distinct from the canonical dataset")

    if not args.input.exists():
        sys.exit(f"[error] missing input: {args.input}")
    # key is checked lazily so fully-cached reruns (zero API calls) need none

    items = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit is not None:
        items = items[: args.limit]

    caches = run(items, locales, args.cache_dir, args.workers)
    records = assemble(items, caches)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] wrote {args.output} ({len(records)} items)")

    if _usage["calls"]:
        print(
            f"[openai] {_usage['calls']} API calls | "
            f"prompt={_usage['prompt']:,} + completion={_usage['completion']:,}"
            f" = {_usage['prompt'] + _usage['completion']:,} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
