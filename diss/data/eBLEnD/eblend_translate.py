"""Build the multilingual extended-BLEnD MCQ: reuse human SAQ questions, LLM-translate options.

Questions come from the SAQ (track_1_saq_input-2.tsv); missing translations prevent
publication. Options are LLM-translated (OpenAI gpt-5.5), one call per
(locale, org_id), with the question's human SAQ answers supplied as a candidate glossary.
The BLEnD annotation JSON is NOT used: its `answers`/`en_answers` lists are independently
collected, not positionally aligned, so any English<->local pairing from them is unreliable.

  python eblend_translate.py --locales ko-KR --limit 50 --output /tmp/eblend_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

BASE = Path(__file__).resolve().parent

DEFAULT_TOPIC_CSV = BASE / "eblend_topic.csv"
DEFAULT_SAQ_QUESTIONS = BASE / "track_1_saq_input-2.tsv"
DEFAULT_SAQ_ANSWERS = BASE / "final_1_saq_reference.tsv"
DEFAULT_CACHE_DIR = BASE / "translation_cache_gpt"
DEFAULT_OUTPUT = BASE / "eblend_final.csv"

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-5.5"
OPENAI_REASONING_EFFORT = "low"
OPENAI_TIMEOUT = 180
LLM_RETRIES = 2
OPTION_KEYS = ("A", "B", "C", "D")
ENGLISH_LOCALE = "en-US"

# locale -> language name, local country name, SAQ country code, SAQ local prefix.
LOCALES = {
    "ko-KR": dict(lang="Korean", country="대한민국", cc="KR", saq="ko"),
    "es-ES": dict(lang="Spanish", country="España", cc="ES", saq="es"),
    "zh-CN": dict(lang="Chinese (Simplified)", country="中国", cc="CN", saq="zh"),
    "ar-SA": dict(lang="Arabic", country="السعودية", cc="SA", saq="ar"),
    "ja-JP": dict(lang="Japanese", country="日本", cc="JP", saq="ja"),
    "fr-FR": dict(lang="French", country="France", cc="FR", saq="fr"),
}


def load_prompt(name: str, **kwargs) -> str:
    """Load {name}.txt and substitute $placeholders (Template, so JSON braces {} pass through)."""
    return Template((BASE / f"{name}.txt").read_text(encoding="utf-8")).substitute(
        **kwargs
    )


_WS_RE = re.compile(r"\s+")
_PAREN_INSTR_RE = re.compile(
    r"\s*\([^()]*(?:provide|numeral|only|e\.g\.|format)[^()]*\)\s*$", re.I
)
_NUMERIC_ONLY_RE = re.compile(r"^\d+(?:\.\d+)?$")
_TIME_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}$")


def norm(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def strip_instruction(question: str) -> str:
    """Drop a trailing parenthetical answer-format instruction from a question."""
    return _PAREN_INSTR_RE.sub("", question).strip()


def is_passthrough(value: str) -> bool:
    """True for content kept verbatim: empty string, bare number, or bare HH:MM time."""
    v = value.strip()
    return (
        v == ""
        or _NUMERIC_ONLY_RE.fullmatch(v) is not None
        or _TIME_ONLY_RE.fullmatch(v) is not None
    )


_JUNK_ANSWERS = {
    "no answer",
    "not applicable",
    "n/a",
    "na",
    "none",
    "unknown",
    "idk",
    "i don't know",
    "-",
}


def is_junk(value: str) -> bool:
    """True for SAQ non-answers to keep out of the reference glossary."""
    v = value.strip().lower()
    return v in _JUNK_ANSWERS or "no answer" in v or "not applicable" in v


class Reference:
    """SAQ questions and answers, indexed for the resolution cascade."""

    def __init__(self, saq_questions: Path, saq_answers: Path):
        self.saq_text: dict[str, str] = {}
        with saq_questions.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.saq_text[r["id"]] = r["text"]
        self.saq_ans: dict[str, str] = {}
        with saq_answers.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.saq_ans[r["id"]] = r["answer"]

        # English SAQ question -> NNN, per country code
        self.saq_en_q: dict[str, dict[str, str]] = defaultdict(dict)
        for sid, txt in self.saq_text.items():
            if sid.startswith("en-"):
                cc = sid.split("-")[1].split("_")[0]
                self.saq_en_q[cc][norm(strip_instruction(txt))] = sid.split("_")[1]

    def resolve_question(
        self, locale: str, question_en: str
    ) -> tuple[str | None, str | None]:
        """Return (local question, NNN) from the SAQ, or (None, None)."""
        cc = LOCALES[locale]["cc"]
        nnn = self.saq_en_q.get(cc, {}).get(norm(strip_instruction(question_en)))
        if nnn is None:
            return None, None
        local = self.saq_text.get(f"{LOCALES[locale]['saq']}-{cc}_{nnn}")
        return local, nnn

    def resolve_option(
        self, locale: str, org_id: str, option_en: str
    ) -> tuple[str | None, str]:
        """Numeric/time/empty options pass through verbatim; everything else is LLM-translated."""
        if is_passthrough(option_en):
            return option_en, "passthrough"
        return None, "llm"

    def glossary(self, locale: str, org_id: str, nnn: str | None) -> str:
        """The question's human local-language SAQ answer pool, junk-filtered and deduped.
        A candidate vocabulary, NOT aligned to the English options."""
        if nnn is None:
            return "  (no reference answers available)"
        cc, prefix = LOCALES[locale]["cc"], LOCALES[locale]["saq"]
        raw = self.saq_ans.get(f"{prefix}-{cc}_{nnn}", "")
        pool: list[str] = []
        seen: set[str] = set()
        for a in raw.split(";"):
            a = a.strip()
            if not a or is_junk(a):
                continue
            key = norm(a)
            if key not in seen:
                seen.add(key)
                pool.append(a)
        return "  " + "; ".join(pool) if pool else "  (no reference answers available)"


_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "translation": {"type": "string"},
                },
                "required": ["id", "source", "translation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


_usage = {"prompt": 0, "completion": 0, "calls": 0}
_usage_lock = threading.Lock()
_API_KEY: str | None = None


def _openai_key() -> str:
    """OpenAI API key, read once from ~/.openai_api_key or $OPENAI_API_KEY."""
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
    """One JSON response from gpt-5.5 (strict json_schema); retried LLM_RETRIES times,
    with longer backoff on HTTP 429."""
    body = json.dumps(
        {
            "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "max_completion_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "translations",
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


def llm_translate(
    locale: str,
    question_en: str,
    question_local: str | None,
    glossary: str,
    strings: list[str],
) -> dict[str, str]:
    """Translate uncovered option strings for one item; returns {source: translation}.
    Each string gets a job-local id so output is matched back exactly, regardless of source text.
    """
    cfg = LOCALES[locale]
    items = [{"id": str(i), "source": s} for i, s in enumerate(strings, 1)]
    prompt = load_prompt(
        "prompt_translation",
        language_name=cfg["lang"],
        country_local=cfg["country"],
        locale_id=locale,
        question_en=question_en,
        question_local=question_local or "(not available)",
        glossary=glossary,
        items=json.dumps(items, ensure_ascii=False, indent=2),
    )
    data = _call_openai_json(prompt, _LLM_SCHEMA)
    id_to_src = {it["id"]: it["source"] for it in items}
    out: dict[str, str] = {}
    for t in data.get("translations", []):
        src = id_to_src.get(t.get("id"), t.get("source"))
        if src is not None:
            out[src] = t["translation"]
    return out


def _load_cache(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(path)


def load_topic_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run(rows: list[dict], ref: Reference, cache_dir: Path, workers: int) -> list[dict]:
    """Resolve every field, dispatch one LLM job per (locale, org_id), return records.
    Cache is keyed locale -> org_id -> source, so the same English word in a different
    question is re-translated with its own glossary; context is never lost to global dedup.
    """
    resolved: dict[str, dict] = {}  # row id -> record
    llm_cache: dict[str, dict[str, dict[str, str]]] = {}  # locale->org_id->{src:tr}
    for locale in LOCALES:
        llm_cache[locale] = _load_cache(cache_dir / f"{locale}.json")

    jobs: dict[tuple[str, str], dict] = {}

    for row in rows:
        locale = row["locale"]
        if locale not in LOCALES:
            continue
        org_id, q_en = row["org_id"], row["question"]
        q_local, nnn = ref.resolve_question(locale, q_en)
        if not isinstance(q_local, str) or not q_local.strip():
            raise ValueError(f"Missing SAQ question: {locale}/{org_id}")
        rec = {
            "question": q_local,
            "question_src": "saq",
            "options": {},
            "option_src": {},
            "nnn": nnn,
            "org_id": org_id,
        }

        cached_q = llm_cache[locale].get(org_id, {})
        for k in OPTION_KEYS:
            opt = row[f"option {k}"]
            t, src = ref.resolve_option(locale, org_id, opt)
            rec["options"][k] = t
            rec["option_src"][k] = src
            if src == "llm" and opt not in cached_q:
                job = jobs.get((locale, org_id))
                if job is None:
                    job = jobs[(locale, org_id)] = {
                        "locale": locale,
                        "org_id": org_id,
                        "nnn": nnn,
                        "question_en": q_en,
                        "strings": [],
                    }
                if opt not in job["strings"]:
                    job["strings"].append(opt)
        resolved[row["id"]] = rec

    print(
        f"[resolve] {len(resolved)} items  |  LLM jobs: {len(jobs)} questions  "
        f"(strings: {sum(len(j['strings']) for j in jobs.values())})"
    )

    def _do(job: dict) -> tuple[str, str, dict[str, str]]:
        locale, org_id = job["locale"], job["org_id"]
        q_local, _ = ref.resolve_question(locale, job["question_en"])
        glossary = ref.glossary(locale, org_id, job["nnn"])
        result = llm_translate(
            locale, job["question_en"], q_local, glossary, job["strings"]
        )
        if any(not isinstance(result.get(s), str) or not result[s].strip()
               for s in job["strings"]):
            raise ValueError(f"Incomplete option translations: {locale}/{org_id}")
        return locale, org_id, result

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_do, j): key for key, j in jobs.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                locale, org_id, result = fut.result()
            except Exception as exc:
                print(f"[llm] job {key}: {exc}")
                continue
            llm_cache[locale].setdefault(org_id, {}).update(result)
            done += 1
            if done % 25 == 0:
                for loc in LOCALES:
                    _save_cache(cache_dir / f"{loc}.json", llm_cache[loc])
                print(f"[llm] {done}/{len(jobs)}")
    for loc in LOCALES:
        _save_cache(cache_dir / f"{loc}.json", llm_cache[loc])

    records = []
    for row in rows:
        locale = row["locale"]
        if locale == ENGLISH_LOCALE:
            records.append(_english_record(row))
            continue
        if locale not in LOCALES:
            continue
        rec = resolved[row["id"]]
        cached_q = llm_cache[locale].get(rec["org_id"], {})

        q = rec["question"]
        q_src = rec["question_src"]

        opts, srcs = {}, {}
        for k in OPTION_KEYS:
            t, src = rec["options"][k], rec["option_src"][k]
            if t is None:
                t = cached_q.get(row[f"option {k}"])
                if not isinstance(t, str) or not t.strip():
                    raise ValueError(f"Missing option translation: {locale}/{row['id']}/{k}")
                src = "llm"
            opts[k], srcs[k] = t, src

        records.append(_record(row, q, q_src, opts, srcs))
    return records


_FIELDS = (
    ["id", "lang", "locale", "topic", "source", "org_id", "answer"]
    + ["question_en"]
    + [f"option {k}_en" for k in OPTION_KEYS]
    + ["question"]
    + [f"option {k}" for k in OPTION_KEYS]
    + ["question_src"]
    + [f"option {k}_src" for k in OPTION_KEYS]
)


def _record(row: dict, q: str, q_src: str, opts: dict, srcs: dict) -> dict:
    return {
        "id": row["id"],
        "lang": row["lang"],
        "locale": row["locale"],
        "topic": row["topic"],
        "source": row["source"],
        "org_id": row["org_id"],
        "answer": row["answer"],
        "question_en": row["question"],
        **{f"option {k}_en": row[f"option {k}"] for k in OPTION_KEYS},
        "question": q,
        **{f"option {k}": opts[k] for k in OPTION_KEYS},
        "question_src": q_src,
        **{f"option {k}_src": srcs[k] for k in OPTION_KEYS},
    }


def _english_record(row: dict) -> dict:
    return _record(
        row,
        row["question"],
        "english_source",
        {k: row[f"option {k}"] for k in OPTION_KEYS},
        {k: "english_source" for k in OPTION_KEYS},
    )


def write_csv(records: list[dict], output: Path) -> None:
    records.sort(key=lambda r: (r["locale"], r["id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    prov = Counter()
    for r in records:
        prov[r["question_src"]] += 1
        for k in OPTION_KEYS:
            prov[r[f"option {k}_src"]] += 1
    print(f"[done] wrote {output} ({len(records)} rows)")
    print("[provenance] " + "  ".join(f"{k}={v}" for k, v in sorted(prov.items())))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=DEFAULT_TOPIC_CSV)
    p.add_argument("--saq-questions", type=Path, default=DEFAULT_SAQ_QUESTIONS)
    p.add_argument("--saq-answers", type=Path, default=DEFAULT_SAQ_ANSWERS)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--locales",
        nargs="*",
        default=None,
        help=f"Subset of {list(LOCALES)} (default: all)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Max items per locale (for smoke runs)"
    )
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.locales is not None and not args.locales:
        p.error("--locales requires at least one locale")
    keep = set(args.locales) if args.locales is not None else (set(LOCALES) | {ENGLISH_LOCALE})
    unknown = keep - (set(LOCALES) | {ENGLISH_LOCALE})
    if unknown:
        p.error(f"unknown locales: {', '.join(sorted(unknown))}")
    canonical_output = args.output.resolve() == DEFAULT_OUTPUT.resolve() or (
        args.output.exists() and DEFAULT_OUTPUT.exists()
        and args.output.samefile(DEFAULT_OUTPUT)
    )
    if (args.limit is not None or args.locales is not None) and canonical_output:
        p.error("--limit/--locales requires --output distinct from the canonical dataset")

    for path in (args.input, args.saq_questions, args.saq_answers):
        if not path.exists():
            sys.exit(f"[error] missing input: {path}")
    # Key is read lazily, so a fully-cached re-run makes zero API calls and needs no key.
    ref = Reference(args.saq_questions, args.saq_answers)
    rows = load_topic_csv(args.input)

    if args.limit is not None:
        count_per_locale: Counter = Counter()
        limited = []
        for r in rows:
            if r["locale"] in keep and count_per_locale[r["locale"]] < args.limit:
                count_per_locale[r["locale"]] += 1
                limited.append(r)
        rows = limited
    else:
        rows = [r for r in rows if r["locale"] in keep]

    records = run(rows, ref, args.cache_dir, args.workers)
    write_csv(records, args.output)
    if _usage["calls"]:
        print(
            f"[openai] {_usage['calls']} API calls | "
            f"prompt={_usage['prompt']:,} + completion={_usage['completion']:,}"
            f" = {_usage['prompt'] + _usage['completion']:,} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
