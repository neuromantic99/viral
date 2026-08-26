from typing import Dict

SESSIONS_KEEP: Dict[str, Dict[str, str | None]] = {
    # Imaging in unsupervised of poor quality, don't analyse
    "JB011": {
        "unsupervised": None,
        "learning": "2024-10-25",
        "learned": "2024-10-30",
    },
    "JB014": {  # LOOKS GOOD
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-04",
    },
    "JB015": {
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-19",
    },
    "JB016": {
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-05",
    },
    "JB018": {
        "unsupervised": "2024-11-20",
        "learning": "2024-11-28",
        "learned": "2024-12-03",
    },
    "JB019": {
        "unsupervised": "2024-11-19",
        "learning": "2024-11-20",
        "learned": "2024-11-22",
    },
    "JB020": {
        "unsupervised": "2024-11-19",
        "learning": "2024-11-20",
        "learned": "2024-11-22",
    },
    "JB021": {
        "unsupervised": "2024-11-29",
        "learning": "2024-12-06",
        "learned": "2024-12-09",
    },
    "JB022": {
        "unsupervised": "2024-12-04",
        "learning": "2024-12-10",
        "learned": "2024-12-12",
    },
    "JB026": {
        "unsupervised": "2024-12-10",
        "learning": "2024-12-13",
        "learned": "2024-12-15",
    },
    "JB027": {
        "unsupervised": "2024-12-10",
        "learning": "2024-12-11",
        "learned": "2024-12-15",
    },
    "JB030": {
        "unsupervised": "2025-03-07",
        "learning": "2025-03-13",
        "learned": "2025-03-14",
    },
    # Imaging was ok for the first few days but then degraded to become not usable
    "JB031": {"unsupervised": "2025-03-07", "learning": "2025-03-12", "learned": None},
    "JB033": {
        "unsupervised": "2025-03-13",
        "learning": "2025-03-17",
        "learned": "2025-03-19",
    },
    "JB034": {
        "unsupervised": "2025-07-04",
        "learning": "2025-07-07",
        "learned": "2025-07-08",
    },
    "JB035": {
        "unsupervised": "2025-07-04",
        "learning": "2025-07-08",
        "learned": "2025-07-11",
    },
    "JB036": {
        "unsupervised": "2025-07-05",
        "learning": "2025-07-07",
        "learned": "2025-07-08",
    },
    "J030": {
        "unsupervised": "2026-05-08",
        "learning": "2026-05-13",
        "learned": "2026-05-15",
    },
    "J031": {
        "unsupervised": "2026-05-08",
        "learning": "2026-05-18",
        "learned": "2026-05-20",
    },
    "J032": {
        "unsupervised": "2026-05-09",
        "learning": "2026-05-15",
        "learned": "2026-05-18",
    },
    "J034": {
        "unsupervised": "2026-06-10",
        "learning": "2025-06-16",
        "learned": "2026-06-18",
    },
    "J035": {
        "unsupervised": "2026-06-10",
        "learning": "2026-02-12",
        "learned": "2026-06-15",
    },
    "J037": {
        "unsupervised": "2026-07-10",
        "learning": "2026-07-14",
        "learned": "2026-07-16",
    },
    "J038": {
        "unsupervised": "2026-07-10",
        "learning": "2026-07-14",
        "learned": "2026-07-15",
    },
}


# 26 is ok, 27 dont use,
# 30 is ok 33 is good
# 32 don't use
