import re
import math
import string
import hashlib
import random
import requests
import os
import numpy as np
import streamlit as st
from pathlib import Path

# ── Load password lists from files ──────────────────────────────────────────

def _load_lines(filename: str) -> set[str]:
    path = Path(__file__).parent / filename
    if not path.exists():
        return set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        return {line.strip().lower() for line in f if line.strip()}

@st.cache_data(show_spinner=False)
def load_common_passwords() -> set[str]:
    return _load_lines("common_weak_passwords.txt") | _load_lines("rockyou.txt")

@st.cache_data(show_spinner=False)
def load_common_words() -> set[str]:
    return _load_lines("common_words.txt")

# ── Entropy & charset detection ──────────────────────────────────────────────

def _charset_size(password: str) -> int:
    size = 0
    if any(c.islower() for c in password):
        size += 26
    if any(c.isupper() for c in password):
        size += 26
    if any(c.isdigit() for c in password):
        size += 10
    if any(c in string.punctuation for c in password):
        size += 32
    return max(size, 1)


def calculate_entropy(password: str) -> float:
    cs = _charset_size(password)
    return round(len(password) * math.log2(cs), 2)


def estimate_crack_time(password: str) -> tuple[str, str]:
    """Returns (human-readable label, severity bucket)."""
    entropy = calculate_entropy(password)
    guesses_per_second = 1e10  # modern GPU cluster baseline
    seconds = (2 ** entropy) / guesses_per_second

    if seconds < 1:
        return "instantly", "critical"
    if seconds < 60:
        return f"{seconds:.1f} seconds", "critical"
    if seconds < 3600:
        return f"{seconds/60:.1f} minutes", "critical"
    if seconds < 86_400:
        return f"{seconds/3600:.1f} hours", "weak"
    if seconds < 2_592_000:
        return f"{seconds/86400:.1f} days", "weak"
    if seconds < 31_536_000:
        return f"{seconds/2_592_000:.1f} months", "fair"
    if seconds < 3_153_600_000:
        return f"{seconds/31_536_000:.1f} years", "strong"
    return f"{seconds/3_153_600_000:.0f}+ centuries", "very strong"


# ── Pattern detection ────────────────────────────────────────────────────────

_KEYBOARD_WALKS = [
    "qwertyuiop", "asdfghjkl", "zxcvbnm",
    "1234567890", "qwerty", "azerty",
]

def _has_keyboard_walk(pw: str, min_len: int = 4) -> bool:
    pw_lower = pw.lower()
    for walk in _KEYBOARD_WALKS:
        for i in range(len(walk) - min_len + 1):
            if walk[i:i+min_len] in pw_lower:
                return True
    return False


def _has_repeated_chars(pw: str, threshold: int = 3) -> bool:
    for i in range(len(pw) - threshold + 1):
        if len(set(pw[i:i+threshold])) == 1:
            return True
    return False


def _has_common_word(pw: str, words: set[str]) -> bool:
    pw_lower = pw.lower()
    return any(w in pw_lower for w in words if len(w) >= 4)


def _leet_normalize(pw: str) -> str:
    leet = {"@": "a", "4": "a", "3": "e", "1": "i", "!": "i",
            "0": "o", "5": "s", "$": "s", "7": "t", "+": "t"}
    return "".join(leet.get(c, c) for c in pw.lower())


# ── HaveIBeenPwned (k-anonymity) ─────────────────────────────────────────────

def check_hibp(password: str) -> int:
    """Returns the number of times the password appeared in known breaches (0 = clean)."""
    try:
        sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
        prefix, suffix = sha1[:5], sha1[5:]
        resp = requests.get(
            f"https://api.pwnedpasswords.com/range/{prefix}",
            timeout=3,
            headers={"Add-Padding": "true"},
        )
        if resp.status_code != 200:
            return -1  # API unavailable
        for line in resp.text.splitlines():
            h, count = line.split(":")
            if h == suffix:
                return int(count)
        return 0
    except Exception:
        return -1


# ── Strength scoring ─────────────────────────────────────────────────────────

def password_strength(password: str, common_passwords: set, common_words: set) -> dict:
    if not password:
        return {"score": 0, "label": "Empty", "color": "#6b7280", "warnings": []}

    warnings = []
    entropy = calculate_entropy(password)
    _, time_bucket = estimate_crack_time(password)

    # Base score from entropy (0–10)
    score = min(entropy / 7, 10)

    # Bonuses
    if any(c.isupper() for c in password):
        score += 0.5
    if any(c.isdigit() for c in password):
        score += 0.5
    if any(c in string.punctuation for c in password):
        score += 1.0
    if len(password) >= 16:
        score += 1.0
    if len(password) >= 20:
        score += 0.5

    # Penalties
    if password.lower() in common_passwords:
        score -= 5
        warnings.append("This is an extremely common password — it will be guessed immediately.")
    if _leet_normalize(password) in common_passwords:
        score -= 3
        warnings.append("Leet-speak substitution detected — easily cracked by modern tools.")
    if _has_keyboard_walk(password):
        score -= 2
        warnings.append("Keyboard walk pattern detected (e.g., 'qwerty', '1234').")
    if _has_repeated_chars(password):
        score -= 1.5
        warnings.append("Repeated characters weaken the password.")
    if _has_common_word(password, common_words):
        score -= 1
        warnings.append("Contains a common dictionary word — try a passphrase or substitute characters.")
    if len(password) < 8:
        score -= 2
        warnings.append("Too short — use at least 12 characters.")
    elif len(password) < 12:
        warnings.append("Consider using 12+ characters for better security.")

    # Clamp with time-bucket ceiling
    ceilings = {"critical": 2, "weak": 4, "fair": 6, "strong": 8, "very strong": 10}
    score = min(score, ceilings.get(time_bucket, 10))
    score = round(max(0.0, min(10.0, score)), 1)

    if score <= 2:
        label, color = "Very Weak", "#ef4444"
    elif score <= 4:
        label, color = "Weak", "#f97316"
    elif score <= 6:
        label, color = "Fair", "#eab308"
    elif score <= 8:
        label, color = "Strong", "#22c55e"
    else:
        label, color = "Very Strong", "#06b6d4"

    return {"score": score, "label": label, "color": color, "warnings": warnings}


# ── Suggestions ───────────────────────────────────────────────────────────────

def generate_suggestions(password: str, analysis: dict) -> list[str]:
    tips = []
    if len(password) < 12:
        tips.append("Increase length to at least 12 characters (16+ is ideal).")
    if not any(c.isupper() for c in password):
        tips.append("Add uppercase letters (A–Z).")
    if not any(c.isdigit() for c in password):
        tips.append("Include at least one digit (0–9).")
    if not any(c in string.punctuation for c in password):
        tips.append("Use special characters like !@#$%^&*() to expand the character set.")
    if not tips and analysis["score"] < 8:
        tips.append("Extend your password length and avoid predictable patterns.")
    return tips


# ── Password generation ───────────────────────────────────────────────────────

_ADJECTIVES = ["swift", "quantum", "lunar", "neon", "arctic", "amber", "cobalt", "prism"]
_NOUNS = ["cipher", "falcon", "nexus", "storm", "beacon", "delta", "orbit", "forge"]

def _memorable_passphrase() -> str:
    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    num = random.randint(10, 99)
    sym = random.choice("!@#$%&*")
    return f"{adj.capitalize()}{noun.capitalize()}{num}{sym}"


def generate_strong_passwords() -> list[dict]:
    passwords = []

    # 1: Memorable passphrase
    passwords.append({
        "label": "Memorable Passphrase",
        "value": _memorable_passphrase(),
    })

    # 2: Random high-entropy
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    passwords.append({
        "label": "High-Entropy Random",
        "value": "".join(random.choice(chars) for _ in range(20)),
    })

    # 3: Alphanumeric (no symbols, safer for sites with restrictions)
    alphanum = string.ascii_letters + string.digits
    passwords.append({
        "label": "Alphanumeric (No Symbols)",
        "value": "".join(random.choice(alphanum) for _ in range(22)),
    })

    return passwords


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def _color_bar(score: float, color: str):
    pct = score / 10 * 100
    st.markdown(
        f"""
        <div style="background:#1f2937;border-radius:6px;height:10px;width:100%;">
            <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;
                        transition:width 0.4s ease;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Password Strength Analyzer",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom dark-theme CSS
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-box {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }
    .char-pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; margin: 2px;
        border: 1px solid;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔐 Password Strength Analyzer")
st.markdown("Check how strong your password is — and get better alternatives.")
st.divider()

# Load data once
common_passwords = load_common_passwords()
common_words = load_common_words()

# Input
col_input, col_toggle = st.columns([4, 1])
with col_input:
    password = st.text_input(
        "Enter password",
        type="password",
        placeholder="Type a password to analyze…",
        label_visibility="collapsed",
    )
with col_toggle:
    show_plain = st.checkbox("Show", value=False)

if show_plain and password:
    st.code(password, language=None)

if not password:
    st.info("Enter a password above to begin the analysis.")
    st.stop()

# ── Analysis ──────────────────────────────────────────────────────────────────
analysis = password_strength(password, common_passwords, common_words)
crack_time_str, _ = estimate_crack_time(password)
entropy = calculate_entropy(password)

# Score bar
st.markdown(f"### Strength: **{analysis['label']}** — {analysis['score']}/10")
_color_bar(analysis["score"], analysis["color"])
st.write("")

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Entropy", f"{entropy} bits")
m2.metric("Length", len(password))
m3.metric("Crack Time", crack_time_str)
m4.metric("Score", f"{analysis['score']}/10")

st.divider()

# Character composition
st.markdown("#### Character Composition")
char_types = {
    "Lowercase": sum(1 for c in password if c.islower()),
    "Uppercase": sum(1 for c in password if c.isupper()),
    "Digits": sum(1 for c in password if c.isdigit()),
    "Symbols": sum(1 for c in password if c in string.punctuation),
}
cols = st.columns(4)
type_colors = {"Lowercase": "#06b6d4", "Uppercase": "#8b5cf6", "Digits": "#22c55e", "Symbols": "#f59e0b"}
for col, (ctype, count) in zip(cols, char_types.items()):
    with col:
        pct = count / len(password) * 100
        st.markdown(
            f"""<div class="metric-box">
                <div class="metric-value" style="color:{type_colors[ctype]}">{count}</div>
                <div class="metric-label">{ctype} ({pct:.0f}%)</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# Warnings + suggestions
if analysis["warnings"]:
    st.markdown("#### ⚠️ Warnings")
    for w in analysis["warnings"]:
        st.warning(w)

suggestions = generate_suggestions(password, analysis)
if suggestions:
    st.markdown("#### 💡 How to Improve")
    for tip in suggestions:
        st.markdown(f"- {tip}")

# HaveIBeenPwned check
st.divider()
with st.expander("🔍 Check Against Known Breaches (HaveIBeenPwned)"):
    st.caption(
        "Your password is never sent. We use k-anonymity: only the first 5 characters "
        "of its SHA-1 hash are transmitted."
    )
    if st.button("Run Breach Check"):
        with st.spinner("Checking…"):
            count = check_hibp(password)
        if count == -1:
            st.warning("Could not reach the HaveIBeenPwned API. Check your internet connection.")
        elif count == 0:
            st.success("Not found in any known data breaches.")
        else:
            st.error(f"Found in **{count:,}** known data breaches. Change this password immediately.")

# Suggested alternatives
st.divider()
st.markdown("#### 🔑 Stronger Alternatives")
st.caption("Generated locally — none of these are stored or transmitted.")

for pwd in generate_strong_passwords():
    col_label, col_val, col_btn = st.columns([2, 3, 1])
    with col_label:
        st.markdown(f"**{pwd['label']}**")
    with col_val:
        st.code(pwd["value"], language=None)
    with col_btn:
        st.button("Copy", key=pwd["label"], help="Select the code block above and copy it")
