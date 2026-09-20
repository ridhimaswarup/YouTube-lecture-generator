import re
from collections import Counter

import gradio as gr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- Load the AI models (first run downloads them) ----------

print("Loading models, please wait...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedder)

SUM_NAME = "google/flan-t5-base"

tok = AutoTokenizer.from_pretrained(SUM_NAME)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_NAME)

print("Models ready!")


# Remembers finished results, so the same link is instant next time
CACHE = {}


# Words that should never become chapter titles or key concepts
STOP = set(ENGLISH_STOP_WORDS) | {
    "let", "discuss", "called", "students", "hello", "section", "previous",
    "learned", "earlier", "known", "like", "also", "many", "much", "okay",
    "going", "know", "want", "different",
}


# ---------- Small text helpers ----------

def clean(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- Transcript correction ----------

COMMON_ASR_FIXES = {
    "oracles": "atria",
    "oracle": "atrium",
    "auricles": "atria",
    "auricle": "atrium",
    "ventricle's": "ventricles",
    "vves": "valves",
}


def correct_asr_errors(text):
    text = clean(text)

    for wrong, right in COMMON_ASR_FIXES.items():
        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            right,
            text,
            flags=re.IGNORECASE
        )

    return text


# ---------- Remove lecture filler ----------

FILLER_PATTERNS = [
    r"\bhello everyone\b[,.]?",
    r"\bhello students\b[,.]?",
    r"\bhi everyone\b[,.]?",
    r"\bwelcome back\b[,.]?",
    r"\bgood morning everyone\b[,.]?",
    r"\bgood afternoon everyone\b[,.]?",
    r"\bso today we (?:are going to|will|shall)\b",
    r"\btoday we are going to\b",
    r"\bin our previous section\b",
    r"\bin our previous chapter\b",
    r"\bin our previous class\b",
    r"\bin the previous section\b",
    r"\bin the previous chapter\b",
    r"\bin the previous class\b",
    r"\bas we learned earlier\b",
    r"\bas we have learned\b",
    r"\bwe have learned\b",
    r"\bwe learned\b",
    r"\byou have already learned\b",
    r"\bas you already know\b",
    r"\bas you know\b",
    r"\blet us discuss\b",
    r"\blet's discuss\b",
    r"\bokay\b",
    r"\bok\b",
]


def remove_lecture_filler(text):
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tidy(text):
    return re.sub(r"\s+([.,!?;:])", r"\1", text).strip()


def polish(text):
    text = tidy(clean(text))

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    text = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text
    )

    text = text[0].upper() + text[1:]

    text = re.sub(r"([.!?])\1+", r"\1", text)

    if text[-1] not in ".!?":
        text += "."

    return text


# ---------- Convert AI output into bullet points ----------

def make_bullets(text):
    lines = text.splitlines()
    bullets = []

    for line in lines:
        line = line.strip()

        line = re.sub(r"^[•●▪◦*-]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)

        if line:
            line = polish(line)
            bullets.append(line)

    if not bullets and text.strip():
        bullets = [polish(text.strip())]

    return "".join(
        f"<li>{bullet}</li>"
        for bullet in bullets
    )


# ---------- Step A: get the video id from a link ----------

def get_video_id(url):
    url = url.strip().replace(" ", "")

    match = re.search(
        r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})",
        url
    )

    return match.group(1) if match else url


# ---------- Step B: download the transcript ----------

def get_transcript(url):
    api = YouTubeTranscriptApi()

    fetched = api.fetch(
        get_video_id(url),
        languages=["en", "hi"]
    )

    return [
        {
            "text": correct_asr_errors(s.text),
            "start": s.start
        }
        for s in fetched
    ]


# ---------- Step C: cut the transcript into chapters ----------

def chunk_by_time(transcript, window=90):
    if not transcript:
        return []

    chunks = []
    current = []

    start = transcript[0]["start"]

    for seg in transcript:

        if seg["start"] - start >= window and current:

            chunks.append({
                "start": start,
                "text": " ".join(current)
            })

            current = []
            start = seg["start"]

        piece = remove_lecture_filler(
            correct_asr_errors(seg["text"])
        )

        if piece:
            current.append(piece)

    if current:

        chunks.append({
            "start": start,
            "text": " ".join(current)
        })

    if len(chunks) > 1 and len(chunks[-1]["text"].split()) < 40:

        last = chunks.pop()

        chunks[-1]["text"] += " " + last["text"]

    return chunks


# ---------- Step D: AI-written notes ----------

def summarize_long(text, piece_words=300):

    words = text.split()

    pieces = [
        words[i:i + piece_words]
        for i in range(0, len(words), piece_words)
    ]

    if len(pieces) > 1 and len(pieces[-1]) < 80:

        last = pieces.pop()

        pieces[-1] += last

    summaries = []

    for piece in pieces:

        prompt = f"""
Turn this lecture transcript into complete study notes for a student.

IMPORTANT:
- Write 4 to 6 bullet points.
- Cover the important facts, definitions, structures, functions and relationships mentioned in the transcript.
- Do not make the notes unnecessarily short.
- Do not copy the transcript word-for-word.
- Rewrite unclear sentences into proper, simple English.
- Remove greetings, introductions, sign-offs and conversational filler.
- Remove statements about previous sections, previous classes or what students have already learned.
- Remove repetition.
- Correct obvious speech-to-text errors using the surrounding context.
- If a word is incorrectly transcribed, infer the correct word from the subject and sentence.
- Preserve important scientific terms and factual details.
- Do not invent information that is not present in the transcript.
- Use proper grammar, spelling, capitalization and punctuation.
- Each bullet should contain a useful piece of information for revision.
- Output ONLY the bullet points.

Transcript:
{" ".join(piece)}

Study notes:
"""

        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        ids = sum_model.generate(
            **inputs,
            max_new_tokens=180,
            num_beams=2,
            no_repeat_ngram_size=3,
            do_sample=False,
            early_stopping=True
        )

        result = tok.decode(
            ids[0],
            skip_special_tokens=True
        )

        if result:
            summaries.append(result.strip())

    return "\n".join(summaries)


# ---------- Step E: keywords ----------

def candidate_phrases(text):

    words = re.findall(r"[a-z]+", text.lower())

    found = Counter()

    for n in (1, 2):

        for i in range(len(words) - n + 1):

            gram = words[i:i + n]

            if any(len(w) < 3 for w in gram):
                continue

            if n == 1 and len(gram[0]) < 4:
                continue

            if gram[0] in STOP or gram[-1] in STOP:
                continue

            if len(set(gram)) < len(gram):
                continue

            found[" ".join(gram)] += 1

    return [p for p, _ in found.most_common(80)]


def chapter_keywords(text):

    cands = candidate_phrases(text)

    if len(cands) < 2:
        return []

    try:

        return kw_model.extract_keywords(
            text,
            candidates=cands,
            use_mmr=True,
            diversity=0.6,
            top_n=8,
        )

    except ValueError:

        return []


def find_common_words(chunks, share=0.6):

    if len(chunks) < 3:
        return set()

    counts = Counter()

    for c in chunks:

        counts.update(
            set(
                re.findall(
                    r"[a-z]{4,}",
                    c["text"].lower()
                )
            )
        )

    return {
        w
        for w, n in counts.items()
        if n >= share * len(chunks)
    }


def make_title(cands, common=frozenset()):

    phrases = [p for p, _ in cands]

    fresh = [
        p
        for p in phrases
        if not any(w in common for w in p.split())
    ]

    chosen = fresh if fresh else phrases

    used = set()
    parts = []

    for p in chosen:

        words = set(p.split())

        if words & used:
            continue

        parts.append(p.title())
        used |= words

        if len(parts) == 2:
            break

    return " & ".join(parts) if parts else "Chapter"


def key_concepts(all_cands, n=8):

    totals = Counter()

    for cands in all_cands:

        for phrase, score in cands:

            totals[phrase] += score * (
                1.3 if " " in phrase else 1.0
            )

    picked = []

    for phrase, _ in totals.most_common():

        if any(
            phrase in q or q in phrase
            for q in picked
        ):
            continue

        picked.append(phrase)

        if len(picked) == n:
            break

    return [
        p.capitalize()
        for p in picked
    ]


# ---------- Overall summary ----------

def make_summary(text):

    if not text or not text.strip():
        return ""

    # Keep the summary source representative of the whole lecture.
    # The chapter notes are already compressed, so we sample content
    # across the complete set instead of allowing the tokenizer to see
    # only the beginning of a long lecture.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    # Keep up to 2 useful note points from each chapter-sized group.
    # This gives the model coverage across the lecture without making
    # the input unnecessarily large.
    selected = []

    for line in lines:
        if len(line.split()) >= 6:
            selected.append(line)

    if not selected:
        selected = lines

    # Keep the input within a fast, predictable size while preserving
    # coverage from the beginning, middle and end.
    max_source_lines = 36

    if len(selected) > max_source_lines:

        positions = [
            round(
                i * (len(selected) - 1)
                / (max_source_lines - 1)
            )
            for i in range(max_source_lines)
        ]

        selected = [
            selected[p]
            for p in positions
        ]

    summary_source = "\n".join(
        f"- {line}"
        for line in selected
    )

    # Let the model decide how many points are actually necessary.
    # A short lecture may need only a few points; a long lecture can
    # use more. The goal is completeness, not an arbitrary number.
    source_words = len(summary_source.split())

    if source_words < 120:
        target_instruction = "Write 3 to 4 bullet points."
    elif source_words < 250:
        target_instruction = "Write 4 to 6 bullet points."
    elif source_words < 450:
        target_instruction = "Write 5 to 7 bullet points."
    else:
        target_instruction = "Write 6 to 9 bullet points."

    prompt = f"""
Create a useful overall study summary from these lecture notes.

{target_instruction}

IMPORTANT RULES:
- Cover the important ideas from the entire set of notes.
- The summary must be sufficient for a student revising the video.
- Do not summarize only the first few points.
- Include important definitions, processes, structures, functions, relationships and factual distinctions when they are present.
- Combine closely related ideas into one bullet when appropriate.
- Do not repeat the same information in different bullets.
- Do not copy the notes word-for-word.
- Rewrite unclear source material into clear, simple English.
- Correct obvious speech-to-text errors using context.
- Do not invent information that is not present in the notes.
- Do not mention the lecture, speaker, students, previous sections or the process of making the notes.
- Every bullet must be a complete sentence or a complete set of logically connected sentences.
- Never end a bullet halfway through a thought.
- Never end a bullet with a dangling word such as "to", "of", "and", "which", "that", "from", "with" or "the".
- Use proper grammar, spelling, capitalization and punctuation.
- Make every bullet useful for exam revision.
- Output ONLY the bullet points.

Study notes:
{summary_source}

Overall summary:
"""

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = sum_model.generate(
        **inputs,
        max_new_tokens=180,
        num_beams=1,
        no_repeat_ngram_size=3,
        do_sample=False,
        early_stopping=True
    )

    result = tok.decode(
        ids[0],
        skip_special_tokens=True
    ).strip()

    if not result:
        return ""

    # Clean obvious incomplete output before displaying it.
    cleaned_lines = []

    for line in result.splitlines():

        line = line.strip()

        line = re.sub(
            r"^[•●▪◦*-]\s*",
            "",
            line
        )

        line = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            line
        )

        if not line:
            continue

        line = polish(line)

        words = line.rstrip(".!?").split()

        # Drop a final fragment if the model clearly stopped halfway.
        if (
            len(words) >= 5
            and words[-1].lower().rstrip(".,!?") in {
                "to",
                "of",
                "and",
                "which",
                "that",
                "from",
                "with",
                "the",
                "a",
                "an",
                "in",
                "for",
                "by"
            }
        ):
            continue

        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    return "".join(
        f"<li>{line}</li>"
        for line in cleaned_lines
    )


# ---------- Put everything together ----------

def make_chapters(url, progress=gr.Progress()):

    if not url or not url.strip():
        return "Please paste a YouTube link first."

    try:

        video_id = get_video_id(url)

        key = video_id

        if key in CACHE:
            return CACHE[key]

        progress(
            0.05,
            desc="Downloading transcript..."
        )

        transcript = get_transcript(url)

        if not transcript:
            return "No transcript was found for this video."

        chunks = chunk_by_time(transcript)

        if not chunks:
            return "Could not create chapters from the transcript."

        common = find_common_words(chunks)

        all_keywords = []
        chapter_blocks = []
        chapter_notes = []

        for i, c in enumerate(chunks):

            progress(
                0.1 + 0.75 * i / len(chunks),
                desc=f"Writing chapter {i + 1} of {len(chunks)}..."
            )

            clean_chapter_text = remove_lecture_filler(
                correct_asr_errors(c["text"])
            )

            cands = chapter_keywords(
                clean_chapter_text
            )

            all_keywords.append(cands)

            title = make_title(
                cands,
                common
            )

            if title == "Chapter":
                title = "Key Concepts"

            body = summarize_long(
                clean_chapter_text
            )

            chapter_notes.append(body)

            body_html = make_bullets(body)

            chapter_blocks.append(
                f'<div class="chapter-card">'
                f'<div class="chapter-number">'
                f'{i + 1}/{len(chunks)}'
                f'</div>'
                f'<h3>{title}</h3>'
                f'<div class="chapter-notes">'
                f'<ul>{body_html}</ul>'
                f'</div>'
                f'</div>'
            )

        concepts = key_concepts(
            all_keywords
        )

        progress(
            0.9,
            desc="Creating summary..."
        )

        summary = make_summary(
            "\n".join(chapter_notes)
        )

        parts = [
            f'<div class="notes-header">'
            f'<span>📋 AI Notes</span>'
            f'<span>{len(chunks)} sections</span>'
            f'</div>',
            *chapter_blocks,
        ]

        if concepts:

            parts += [
                '<div class="section-block">',
                "<h2>🔑 Key Concepts</h2>",
                '<div class="concepts">',
                " ".join(
                    f"<code>{k}</code>"
                    for k in concepts
                ),
                "</div>",
                "</div>",
            ]

        if summary:

            parts += [
                '<div class="section-block summary-block">',
                "<h2>📝 Summary</h2>",
                f'<div class="summary-text">'
                f'<ul>{summary}</ul>'
                f'</div>',
                "</div>",
            ]

        progress(
            1,
            desc="Done!"
        )

        result = "\n".join(parts)

        CACHE[key] = result

        return result

    except Exception as e:

        return (
            f"Something went wrong: {e}\n\n"
            "Check that the link is correct and the video has "
            "captions (look for the CC button on YouTube)."
        )


# ============================================================
# MINIMALIST UI
# ============================================================

CSS = """
/* ---------- Overall page ---------- */

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 32px 24px 50px !important;
    background: #faf9f7 !important;
    font-family: "Inter", "Segoe UI", sans-serif !important;
}


/* ---------- Main heading ---------- */

#header {
    text-align: center;
    padding: 25px 10px 30px;
}

#header h1 {
    color: #26232a;
    font-size: 2.1em;
    font-weight: 650;
    letter-spacing: -0.8px;
    margin: 0 0 8px 0;
}

#header p {
    color: #77717d;
    font-size: 1em;
    font-weight: 400;
    margin: 0;
}


/* ---------- Input area ---------- */

.input-card {
    background: #ffffff;
    border: 1px solid #ebe7ec;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 3px 14px rgba(50, 40, 60, 0.035);
}


/* ---------- Textbox ---------- */

textarea,
input {
    border-radius: 10px !important;
    border: 1px solid #ded9e1 !important;
    background: #ffffff !important;
}

textarea:focus,
input:focus {
    border-color: #a78bba !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 186, 0.12) !important;
}


/* ---------- Labels ---------- */

label {
    color: #49434d !important;
    font-weight: 500 !important;
}


/* ---------- Main button ---------- */

#go-btn {
    border-radius: 10px !important;
    font-size: 0.98em !important;
    font-weight: 600 !important;
    background: #8f719f !important;
    border: none !important;
    box-shadow: none !important;
}

#go-btn:hover {
    background: #7e618e !important;
}


/* ---------- Clear button ---------- */

button {
    border-radius: 10px !important;
}


/* ---------- Results ---------- */

#result {
    background: #ffffff !important;
    border: 1px solid #ebe7ec !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    min-height: 260px;
    box-shadow: 0 3px 14px rgba(50, 40, 60, 0.035);
}


/* ---------- AI Notes header ---------- */

.notes-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f7f7f7;
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 18px;
    color: #403847;
    font-size: 1.05em;
    font-weight: 600;
}

.notes-header span:last-child {
    color: #77717d;
    font-size: 0.9em;
    font-weight: 500;
}


/* ---------- Chapter cards ---------- */

.chapter-card {
    position: relative;
    background: #ffffff;
    border: 1px solid #ebe7ec;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 3px 8px rgba(50, 40, 60, 0.045);
}

.chapter-card h3 {
    color: #29252c;
    font-size: 1.02em;
    font-weight: 650;
    margin: 0 55px 12px 0;
}

.chapter-number {
    position: absolute;
    right: 18px;
    top: 18px;
    color: #9a949e;
    font-size: 0.86em;
}

.chapter-notes {
    color: #69636d;
    line-height: 1.65;
    font-size: 0.94em;
}

.chapter-notes p {
    margin: 0;
}

.chapter-notes ul {
    margin: 0;
    padding-left: 20px;
}

.chapter-notes li {
    margin-bottom: 7px;
}


/* ---------- Sections ---------- */

.section-block {
    background: #ffffff;
    border: 1px solid #ebe7ec;
    border-radius: 16px;
    padding: 18px 22px;
    margin-top: 18px;
    box-shadow: 0 3px 8px rgba(50, 40, 60, 0.045);
}

.section-block h2 {
    color: #29252c;
    font-size: 1.1em;
    font-weight: 650;
    margin: 0 0 12px 0;
}


/* ---------- Key concept tags ---------- */

.concepts {
    line-height: 2.4;
}

#result code {
    background: #f3eef6;
    color: #7a5a8c;
    padding: 3px 11px;
    border-radius: 999px;
    border: 1px solid #e6dcec;
    font-size: 0.88em;
    font-family: inherit;
    margin-right: 2px;
}


/* ---------- Summary ---------- */

.summary-text {
    color: #69636d;
    line-height: 1.65;
    font-size: 0.94em;
}

.summary-text ul {
    margin: 0;
    padding-left: 20px;
}

.summary-text li {
    margin-bottom: 7px;
}


/* ---------- Accordions ---------- */

.gradio-accordion {
    border: 1px solid #ebe7ec !important;
    border-radius: 14px !important;
    background: #ffffff !important;
}


/* ---------- Examples ---------- */

.gradio-examples {
    border-radius: 12px !important;
}


/* ---------- Subtle text ---------- */

.gradio-container .info {
    color: #817b84 !important;
}


/* ---------- Mobile ---------- */

@media (max-width: 700px) {

    .gradio-container {
        padding: 20px 14px 35px !important;
    }

    #header h1 {
        font-size: 1.7em;
    }

    #header p {
        font-size: 0.92em;
    }

    #result {
        padding: 18px !important;
    }

    .chapter-card {
        padding: 16px 18px;
    }
}
"""


EMPTY_TEXT = """
### Your notes will appear here

Paste a link and press **Make Chapters**.
"""


# ---------- The web page ----------

with gr.Blocks(
    title="Lecture Chapter Generator"
) as demo:

    gr.HTML(
        """
        <div id="header">
            <h1>Lecture Chapter Generator</h1>
            <p>
                Turn a YouTube lecture into AI notes,
                key concepts and a summary.
            </p>
        </div>
        """
    )

    with gr.Accordion(
        "How to use",
        open=True
    ):

        gr.Markdown(
            """
1. **Paste** a YouTube lecture link.
2. **Click Make Chapters.** You get structured AI notes, key concepts and a summary.
            """
        )

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes="input-card"
        ):

            url = gr.Textbox(
                label="YouTube lecture link",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1,
            )

            with gr.Row():

                clear_btn = gr.Button(
                    "Clear"
                )

                go_btn = gr.Button(
                    "Make Chapters",
                    variant="primary",
                    elem_id="go-btn"
                )

            gr.Examples(
                examples=[
                    [
                        "https://www.youtube.com/watch?v=ML2WX84gsGE"
                    ]
                ],
                inputs=[
                    url
                ],
                label="Try an example",
                cache_examples=False,
            )

        with gr.Column(scale=1):

            out = gr.Markdown(
                EMPTY_TEXT,
                elem_id="result"
            )

    with gr.Accordion(
        "Tips & limitations",
        open=False
    ):

        gr.Markdown(
            """
- Works only on videos that have **captions**.
- Chapters are still created internally from the transcript.
- All chapter text is rewritten by AI into concise study notes.
- Auto-generated captions can mishear words, and the AI attempts to correct them using context.
- Running the same link again is instant.
            """
        )

    # Make Chapters button

    go_btn.click(
        make_chapters,
        [url],
        out
    )

    # Press Enter in URL box

    url.submit(
        make_chapters,
        [url],
        out
    )

    # Clear button

    clear_btn.click(
        lambda: ("", EMPTY_TEXT),
        None,
        [url, out]
    )


# ---------- Run the app ----------

if __name__ == "__main__":

    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink"
        ),
        css=CSS,
    )