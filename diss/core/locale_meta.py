"""Locale/country/per-locale dataset-config single source of truth."""

from __future__ import annotations

LOCALE_TO_COUNTRY: dict[str, str] = {
    "ko": "South Korea",
    "es": "Spain",
    "zh": "China",
    "ar": "Saudi Arabia",
    "ja": "Japan",
    "fr": "France",
    "en": "United States",
}

COUNTRY_TO_LOCALE: dict[str, str] = {v: k for k, v in LOCALE_TO_COUNTRY.items()}

ALL_TARGET_LOCALES: list[str] = list(LOCALE_TO_COUNTRY.keys())
ALL_TARGET_COUNTRIES: list[str] = list(LOCALE_TO_COUNTRY.values())

# Display order for plots/CSVs; distinct from manifest "languages" eval scope.
LOCALE_DISPLAY_ORDER: list[str] = ["ko", "ja", "zh", "fr", "es", "ar", "en"]

# Language–country centricity: monocentric = one dominant country sets the standard culture;
# pluricentric = multiple national standards
LOCALE_CENTRICITY: dict[str, str] = {
    "ko": "monocentric",
    "ja": "monocentric",
    "zh": "pluricentric",
    "es": "pluricentric",
    "ar": "pluricentric",
    "fr": "pluricentric",
    "en": "pluricentric",
}

# Localized country names; outer key = rendering locale, inner key = English
# country name (the 4 extra entries are the distractor pool for ablations).
COUNTRY_NAMES_BY_LOCALE: dict[str, dict[str, str]] = {
    "ko": {
        "South Korea": "대한민국",
        "Spain": "스페인",
        "China": "중국",
        "Saudi Arabia": "사우디아라비아",
        "Japan": "일본",
        "France": "프랑스",
        "United States": "미국",
        "United Kingdom": "영국",
        "Mexico": "멕시코",
        "Indonesia": "인도네시아",
        "Iran": "이란",
    },
    "es": {
        "South Korea": "Corea del Sur",
        "Spain": "España",
        "China": "China",
        "Saudi Arabia": "Arabia Saudita",
        "Japan": "Japón",
        "France": "Francia",
        "United States": "Estados Unidos",
        "United Kingdom": "Reino Unido",
        "Mexico": "México",
        "Indonesia": "Indonesia",
        "Iran": "Irán",
    },
    "zh": {
        "South Korea": "韩国",
        "Spain": "西班牙",
        "China": "中国",
        "Saudi Arabia": "沙特阿拉伯",
        "Japan": "日本",
        "France": "法国",
        "United States": "美国",
        "United Kingdom": "英国",
        "Mexico": "墨西哥",
        "Indonesia": "印度尼西亚",
        "Iran": "伊朗",
    },
    "ar": {
        "South Korea": "كوريا الجنوبية",
        "Spain": "إسبانيا",
        "China": "الصين",
        "Saudi Arabia": "المملكة العربية السعودية",
        "Japan": "اليابان",
        "France": "فرنسا",
        "United States": "الولايات المتحدة",
        "United Kingdom": "المملكة المتحدة",
        "Mexico": "المكسيك",
        "Indonesia": "إندونيسيا",
        "Iran": "إيران",
    },
    "ja": {
        "South Korea": "韓国",
        "Spain": "スペイン",
        "China": "中国",
        "Saudi Arabia": "サウジアラビア",
        "Japan": "日本",
        "France": "フランス",
        "United States": "アメリカ",
        "United Kingdom": "イギリス",
        "Mexico": "メキシコ",
        "Indonesia": "インドネシア",
        "Iran": "イラン",
    },
    "fr": {
        "South Korea": "Corée du Sud",
        "Spain": "Espagne",
        "China": "Chine",
        "Saudi Arabia": "Arabie saoudite",
        "Japan": "Japon",
        "France": "France",
        "United States": "États-Unis",
        "United Kingdom": "Royaume-Uni",
        "Mexico": "Mexique",
        "Indonesia": "Indonésie",
        "Iran": "Iran",
    },
    "en": {
        "South Korea": "South Korea",
        "Spain": "Spain",
        "China": "China",
        "Saudi Arabia": "Saudi Arabia",
        "Japan": "Japan",
        "France": "France",
        "United States": "United States",
        "United Kingdom": "United Kingdom",
        "Mexico": "Mexico",
        "Indonesia": "Indonesia",
        "Iran": "Iran",
    },
}

# Belebele HF config per locale (FLORES-200 codes).
BELEBELE_CONFIG_BY_LOCALE: dict[str, str] = {
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "ja": "jpn_Jpan",
    "fr": "fra_Latn",
    "en": "eng_Latn",
}


CULTURE_COUNTRY_BY_LOCALE = LOCALE_TO_COUNTRY
COUNTRY_NAMES = COUNTRY_NAMES_BY_LOCALE
LOCALE_TO_BELEBELE_CONFIG = BELEBELE_CONFIG_BY_LOCALE
SUPPORTED_LOCALES = set(ALL_TARGET_LOCALES)
