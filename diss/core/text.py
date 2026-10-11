from __future__ import annotations

LOCALE_KEYWORDS: dict[str, tuple[str, str]] = {
    "ko": ("질문:", "정답:"),
    "es": ("Pregunta:", "Respuesta:"),
    "zh": ("问题：", "答案："),
    "ar": ("السؤال:", "الإجابة:"),
    "ja": ("質問:", "答え:"),
    "fr": ("Question :", "Réponse :"),
    "en": ("Question:", "Answer:"),
}

LOCALE_PASSAGE_KEYWORDS: dict[str, str] = {
    "ko": "지문:",  # 지문에 따르면,
    "es": "Texto:",  # De acuerdo con el texto,
    "zh": "文章:",  # 根据这段文字，
    "ar": "النص:",
    "ja": "本文:",  # 文中の内容によると、
    "fr": "Passage :",  # D’après l’extrait,
    "en": "Passage:",  # According to the passage,
}


def build_continuation_context(
    question: str,
    locale: str,
    *,
    passage: str | None = None,
) -> str:
    q_kw, a_kw = LOCALE_KEYWORDS.get(locale, LOCALE_KEYWORDS["en"])
    if passage is not None:
        p_kw = LOCALE_PASSAGE_KEYWORDS.get(locale, LOCALE_PASSAGE_KEYWORDS["en"])
        return f"{p_kw} {passage.strip()}\n{q_kw} {question.strip()}\n{a_kw}"
    return f"{q_kw} {question.strip()}\n{a_kw}"
