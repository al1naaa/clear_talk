"""Build the English technical report from the aggregate JSON files beside this source."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

HERE = Path(__file__).resolve().parent
LANGS = {"ru": "Russian", "kk": "Kazakh", "en": "English"}
CONDS = {
    "clean": "Original",
    "white-10": "White, 10 dB",
    "babble-10": "4-talker babble, 10 dB",
}
SEPS = {"libri": "Libri2Mix", "wham": "WHAM!", "adapted": "Libri2Mix + fitted decoder"}
W, H = A4
M = 17 * mm
WIDTH = W - 2 * M
INK = colors.HexColor("#17323a")
ACCENT = colors.HexColor("#006c66")
MUTED = colors.HexColor("#465b61")
RULE = colors.HexColor("#cbd8d6")
paper = Canvas(
    str(HERE / "ClearTalk-technical-report.pdf"), pagesize=A4, pageCompression=1
)
paper.setTitle("ClearTalk | Noisy speech and two simultaneous voices")
paper.setAuthor("ClearTalk project")
current = json.loads((HERE / "current.json").read_text())
archive = json.loads((HERE / "metrics.json").read_text())
asr = current["asr_validation"]
external = current["noisy_controls"]["systems"][str(asr["selected_step"])]
choice = current["separation_selection"]
test = current["separation_test"]


def pct(value):
    return f"{100 * value:.2f}%"


def ratio(row):
    return f"{pct(row['wer'])} / {pct(row['cer'])}"


def page(section, title):
    global y
    paper.setFillColor(ACCENT)
    paper.setFont("Helvetica-Bold", 10)
    paper.drawString(M, H - M, "CLEARTALK   /   TECHNICAL REPORT")
    paper.setFont("Helvetica", 9)
    paper.drawRightString(W - M, H - M, "21 NOVEMBER 2026")
    paper.setStrokeColor(RULE)
    paper.line(M, H - M - 11, W - M, H - M - 11)
    y = H - M - 38
    para(section.upper(), 9, ACCENT, bold=True)
    para(title, 23, INK, bold=True, gap=13)


def para(text, size=11, color=INK, bold=False, gap=8):
    global y
    style = ParagraphStyle(
        "text",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=size * 1.36,
        textColor=color,
    )
    block = Paragraph(text, style)
    _, height = block.wrap(WIDTH, H)
    y -= height
    block.drawOn(paper, M, y)
    y -= gap


def sub(text):
    para(text, 13, ACCENT, bold=True, gap=5)


def table(header, rows, widths):
    global y
    exact = [WIDTH * w / sum(widths) for w in widths]
    data = [header, *rows]
    cells = []
    for i, row in enumerate(data):
        style = ParagraphStyle(
            "cell",
            fontName="Helvetica-Bold" if i == 0 else "Helvetica",
            fontSize=9.4,
            leading=12,
            textColor=colors.white if i == 0 else INK,
        )
        cells.append([Paragraph(escape(str(cell)), style) for cell in row])
    t = Table(cells, colWidths=exact, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#eef5f3")],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    _, h = t.wrap(WIDTH, H)
    y -= h
    t.drawOn(paper, M, y)
    y -= 14


def finish():
    if y < 45:
        raise ValueError(f"Page {paper.getPageNumber()} overflows: {y:.1f}")
    paper.setFont("Helvetica", 8.5)
    paper.setFillColor(MUTED)
    paper.drawString(
        M, 26, "ClearTalk   /   speech experiments and deployment evidence"
    )
    paper.drawRightString(W - M, 26, str(paper.getPageNumber()))
    paper.showPage()


page("01  /  Findings", "Noisy speech and two simultaneous voices")
para(
    "Evaluation of automatic speech recognition in Russian, Kazakh and English",
    14,
    MUTED,
)
sub("Deployment decision")
para(
    f"The application retains the step-600 multilingual recognizer for Russian and Kazakh, generic Whisper large-v3-turbo for English turns, and the {SEPS[choice['selected']]} separator for simultaneous voices. We compared new training runs with the deployed models on validation recordings before running the independent controls."
)
sub("Three distinct quality questions")
table(
    ["Question", "Evidence", "Measured outcome"],
    [
        [
            "Can new ASR training reduce noisy word errors?",
            "36 FLEURS validation utterances, each rendered three ways.",
            f"Original macro noisy WER {pct(asr['steps']['0']['macro_noisy_wer'])}. None of three new checkpoints passed the deployment rule.",
        ],
        [
            "Can an acoustic improvement survive transcription?",
            "18 validation views of nine bilingual source pairs.",
            "The refitted synthesis decoder added 0.023 dB over Libri2Mix but made one additional word error across 672 reference words.",
        ],
        [
            "What does the selected pipeline return on other recordings?",
            "54 single-speaker controls and 18 bilingual pairs, each evaluated in fixed noise conditions.",
            f"New mixed-language test cpWER {pct(test['all']['cpwer'])}. The separator gained {test['all']['si_sdri_db']:+.2f} dB over the mixture. Pages 3 and 6 give the breakdown.",
        ],
    ],
    [1.04, 1.12, 1.42],
)
sub("Two routes through the application")
para(
    "Conversation mode uses speech detection, automatically routes turns within the three-language vocabulary, and emits speaker, language, time and text. Choosing two simultaneous voices splits a short mono recording first and transcribes both outputs separately. Short English-only separation has a dedicated small.en recognizer adapted to separator artifacts."
)
para(
    "Word error rate can remain high even when the two output waveforms are measurably closer to their sources. The student project therefore reports transcript and acoustic metrics separately.",
    10,
    MUTED,
)
finish()

page("02  /  Experiment 1", "What changed inside Whisper")
para(
    "The incumbent is a 2,129,920-parameter decoder LoRA adaptation (rank 16) of a Kazakh Whisper large-v3-turbo prior. Its earlier training recipe contained 14,400 examples, including 9,000 mixed recordings. Validation selected step 600. The starting large model's mixed-utterance WER fell from 30.87% to 9.78% on 300 earlier validation recordings."
)
sub("Fresh noise-aware training")
para(
    "We reconstructed the native trainable weights by applying 28 merged decoder matrices from the local inference release to the pinned native checkpoint. Re-exporting the entire initialization to CTranslate2 FP16 reproduced the deployed model.bin SHA-256 byte for byte. A fresh AdamW optimizer fit a rank-8 adapter with alpha 16 and dropout 0.05. Its 245,760 trainable parameters cover query/value projections in encoder blocks 30-31 and all four decoder cross-attention blocks."
)
para(
    "Each of 96 updates accumulated six individual examples, two per language. FLEURS supplied 576 distinct training utterances. Half remained clean. The other half received white noise, pink noise, ESC-10 audio or four-talker babble at 3-18 dB SNR. Peak learning rate was 0.00006, with eight warm-up steps and cosine decay. Training used float32 on Apple MPS."
)
sub("Matched validation at production decoding settings")
rows = []
for step, rec in asr["steps"].items():
    low, high = rec["delta_ci95"]
    rows.append(
        [
            "Incumbent" if step == "0" else f"LoRA step {step}",
            pct(rec["macro_noisy_wer"]),
            f"{100 * low:+.2f} to {100 * high:+.2f}",
            "Pass" if rec["clean_safe"] else "Fail",
        ]
    )
table(
    ["Checkpoint", "Macro noisy WER", "Paired 95% delta (pp)", "Clean guard"],
    rows,
    [1.05, 1, 1.25, 0.8],
)
para(
    "Twelve validation utterances per language contributed original, white-noise and babble views. Macro noisy WER weights each of six language/noise cells equally. Promotion required a 10% relative reduction, a paired 95% interval below zero, and clean WER within 0.5 percentage points in every language. The 4,000 bootstrap draws resampled utterances within language and kept both noise views together.",
    10,
)
enh = current["enhancement_validation"]
para(
    f"A separate stationary-noise trial calibrated its trigger on 144 training clips. Frozen MetricGAN+ with a dry-signal blend changed {enh['enhanced_views']} of 108 validation views. Macro noisy WER rose to {pct(enh['macro_noisy_wer'])}, so deployment keeps the original waveform.",
    10,
    MUTED,
)
finish()

page("03  /  Independent single-speaker test", "Errors by language and noise condition")
para(
    "The selected ASR pipeline processed 18 recordings per language. Each original was scored with no additional noise, with white noise at 10 dB SNR and with four-talker babble at 10 dB SNR. All settings, routes and thresholds were fixed on development audio. Environmental sound present in an original recording remains present."
)
rows = []
for code, lang in LANGS.items():
    for c, label in CONDS.items():
        r = external[code + "_" + c]
        rows.append(
            [
                lang,
                label,
                r["reference_words"],
                pct(r["wer"]),
                pct(r["cer"]),
                f"{r['exact_lid']}/{r['n']}",
            ]
        )
table(
    ["Language", "Added noise", "Ref. words", "WER", "CER", "Exact label"],
    rows,
    [0.90, 1.75, 0.8, 0.75, 0.75, 0.84],
)
sub("Automatic language identification")
para(
    "Exact label counts the entire clip as correct when its nonempty output contains only the reference language label. It includes transcription-derived labels and speech detection. Below, the restricted acoustic detector is measured separately. The denominator counts actual detector calls across all three conditions. A clip can trigger several calls or none."
)
det = []
for l, name in LANGS.items():
    rs = [external[l + "_" + c] for c in CONDS]
    good = sum(r["detector_correct"] for r in rs)
    n = sum(r["detector_calls"] for r in rs)
    det.append(
        [
            name,
            f"{good}/{n}",
            pct(good / n),
            f"{sum(r['clips_with_detector_calls'] for r in rs)}/54",
        ]
    )
table(
    ["Language", "Correct calls", "Call accuracy", "Clip views with a call"],
    det,
    [1, 1, 1, 1.5],
)
para(
    "Kazakh controls came from KSD, Russian from Golos far-field and English from LibriSpeech. Language and corpus are confounded. A comparison across languages also compares microphones, utterance styles and corpus annotations. Repeated noise views share a source recording.",
    10,
    MUTED,
)
finish()

page("04  /  Archived multilingual benchmark", "The larger Test D comparison")
para(
    "We rescored saved automatic transcripts for 898 utterances under three conditions, giving 2,694 matched views per system. Reference IDs and all historical condition-level totals were verified. The comparison isolates application changes including speech-region decoding, repetition handling and background-turn filtering. Both sides use the same underlying trained recognizers."
)
hist = {(r["system"], r["language"], r["condition"]): r for r in archive["noise"]}
rows = []
for code, lang in LANGS.items():
    for c, label in CONDS.items():
        a, b = [hist[s, code, c] for s in ("app-before", "app-final")]
        rows.append(
            [
                lang,
                label,
                b["n"],
                ratio(a),
                ratio(b),
                f"{b['exact_language_labels']}/{b['n']}",
            ]
        )
table(
    [
        "Language",
        "Noise",
        "Clips",
        "Before WER / CER",
        "Final WER / CER",
        "Exact label",
    ],
    rows,
    [0.83, 1.30, 0.51, 1.43, 1.43, 0.83],
)
sub("How to interpret this test")
para(
    "Kazakh contributed 600 KSC2 and 100 FLEURS utterances. Russian contributed 100 Golos crowd and 80 FLEURS utterances. English contributed 18 FLEURS utterances. A single pooled number would give Kazakh recordings much greater weight and hide the small English sample."
)
para(
    "The historical noise-trained decoder was rejected on validation. The Test D changes belong to the deployed application settings. The current external controls on page 3 are fresh inference on different source recordings, domains and CPU precision."
)
sub("Scoring convention")
para(
    "WER pools word insertions, deletions and substitutions over reference words. CER pools analogous character edits over non-space reference characters. Multilingual scoring uses Unicode NFKC, lowercase, the project normalization of Russian yo, and consistent punctuation handling. The historical English T1/T2 comparison retains its own English normalizer. Scores are pooled only within a benchmark.",
    10,
)
finish()

page("05  /  Experiment 2", "Training and selecting a separator")
para(
    "SepFormer estimates two source masks at 8 kHz. In the new acoustic run we froze its Libri2Mix analysis encoder and mask estimator, then refit 4,096 synthesis-decoder parameters on 192 three-second FLEURS mixtures. Each language pair contributed 64 mixtures with random level and onset differences. Three quarters contained additional white or pink noise at 3-15 dB."
)
para(
    "A batch of eight optimized negative permutation-invariant SI-SDR with AdamW at 0.0001 for six epochs. On 24 separate short validation mixtures, we chose the highest average improvement subject to a clean-slice drop of at most 0.1 dB."
)
history = current["acoustic_validation"]["history"]
table(
    ["Decoder epoch", "All SI-SDRi", "Clean SI-SDRi", "Noisy SI-SDRi"],
    [
        [e, f"{m['all']:+.3f} dB", f"{m['clean']:+.3f} dB", f"{m['noise']:+.3f} dB"]
        for e, m in history.items()
    ],
    [1, 1, 1, 1],
)
sub("Validation through the complete transcription pipeline")
rows = []
for sep, m in choice["validation"].items():
    r = m["all"]
    rows.append(
        [
            SEPS[sep],
            f"{r['errors']}/{r['words']}",
            pct(r["cpwer"]),
            f"{r['si_sdri_db']:+.3f} dB",
        ]
    )
table(
    ["Separator", "Word edits / words", "cpWER", "SI-SDRi"],
    rows,
    [1.8, 1.2, 0.88, 0.95],
)
para(
    "Each candidate received the same nine bilingual FLEURS pairs twice, once without added noise and once at 10 dB white noise. The ASR remained fixed. Selection minimized pooled cpWER among candidates with mean SI-SDRi at least equal to the original Libri2Mix separator. All three returned two tracks in every validation view."
)
sub("Metrics measure different failure modes")
para(
    "cpWER concatenates output segments per speaker in time order, pads missing or extra streams, and chooses the speaker assignment with the fewest word edits. SI-SDR projects each estimate onto a time-aligned clean stem. We maximize the two-source mean across assignments, subtract the same two stems' mean score against the unprocessed mixture, then average improvements across mixtures.",
    10,
)
finish()

page("06  /  Held-out two-speaker test", "From waveform recovery to transcript quality")
para(
    "Eighteen bilingual source pairs were built from the external controls, six for each pairing. Each utterance appeared in one pair. The second voice began after a quarter of the shorter clip's length. Its relative level cycled through -3, 0 and +3 dB. Each pair was rendered with and without 10 dB white noise. The clean reference stems were used for scoring only."
)
rows = []
for pair, label in [
    ("kk+ru", "Kazakh + Russian"),
    ("ru+en", "Russian + English"),
    ("en+kk", "English + Kazakh"),
]:
    for c in ("clean", "white-10"):
        r = test[pair + "_" + c]
        rows.append(
            [
                label,
                CONDS[c],
                f"{r['errors']}/{r['words']}",
                pct(r["cpwer"]),
                f"{r['si_sdri_db']:+.2f} dB",
                f"{r['two_tracks']}/{r['n']}",
            ]
        )
r = test["all"]
rows.append(
    [
        "All pairs",
        "Both views",
        f"{r['errors']}/{r['words']}",
        pct(r["cpwer"]),
        f"{r['si_sdri_db']:+.2f} dB",
        f"{r['two_tracks']}/{r['n']}",
    ]
)
table(
    ["Source languages", "Condition", "Edits / words", "cpWER", "SI-SDRi", "2 tracks"],
    rows,
    [1.55, 1.03, 0.98, 0.76, 0.92, 0.73],
)
isolated = current.get("isolated_sources")
if isolated:
    vals = [pct(isolated[k]["cpwer"]) for k in ("kk+ru", "ru+en", "en+kk")]
    para(
        "When the same voices are transcribed from their original separate recordings, paired cpWER is "
        + ", ".join(vals)
        + " in the row order above. This diagnostic isolates the cost of mixing and recovering the voices.",
        10,
        MUTED,
    )
sub("English-only comparison on the earlier test sets")
para(
    "The English route uses the same Libri2Mix separator for both recognizers. The original Whisper small.en is compared with a rank-16 LoRA trained on separated speech, including separator artifacts. This run updated 6,488,064 parameters, used 17,632 tracks per epoch and selected step 1,102 by validation. Variation in cpWER here measures the recognizer change."
)
rows = []
for h in sorted(archive["overlap"], key=lambda m: m["n"]):
    rows.append(
        [
            "T1" if h["n"] == 96 else "T2",
            h["n"],
            pct(h["baseline_cpwer"]),
            pct(h["cleartalk_cpwer"]),
            f"{h['si_sdri_db']:+.2f} dB",
        ]
    )
table(
    ["Set", "Pairs", "Pretrained cpWER", "Tuned cpWER", "SI-SDRi"],
    rows,
    [0.6, 0.65, 1.3, 1.17, 0.9],
)
para(
    "T2 uses 9,367 reference words. Its reconstructed mixture matched the archived PCM within one integer level. On a separate 28-clip AMI meeting check, archived WER rose from 29.21% to 34.06% after English adaptation. Natural meetings remain a harder transfer case.",
    10,
    MUTED,
)
finish()

page("07  /  Replication and limitations", "What the evidence supports")
sub("Language and speaker scope")
para(
    "Automatic language routing was tested for Russian, Kazakh and English. The two-source language pairs were also measured individually. Speaker numbers identify streams within a clip. The held-out pairs combine independent utterances and do not capture natural timing, room acoustics, code switching within a single speaker, or conversations with three simultaneous speakers."
)
sub("Control-set integrity")
para(
    "The six exploratory recordings per language and later 18 controls per language were selected by fixed rules. No exploratory ID entered the later set. Kazakh selections group filename prefixes as a proxy for recording origin. Russian reader identities are unavailable. English controls hold out readers from the pilot but come from an older project test manifest. Upstream foundation-model training overlap is unknown."
)
para(
    "All 28 local KSD payloads matched the official ZIP member sizes and CRCs. One member could not be decoded through libsndfile. In one exploratory example the clean hypothesis matched the neighboring official transcript word for word, suggesting an annotation offset. The published exploratory cpWER uses the supplied references: 90/126 errors (71.43%) on six noisy Kazakh-Russian pairs. Excluding that one pair gives 71/106 (66.98%) as a sensitivity check. The flagged recording is outside the final controls. The same exact-neighbor check flagged zero of the 18 final clean Kazakh clips."
)
sub("Reproduction and demo")
para(
    "The accompanying REPRODUCE.md gives exact commands for source checks, restoration, fitting, validation, decoding and scoring. The metrics.json and current.json files contain the aggregate counts underlying these tables. Run the local service, open the two-voice demo, or use the saved offline HTML to play the noisy input and both returned tracks with their unedited transcripts."
)
para("Public project and source records", 10, ACCENT, bold=True)
for label, url in [
    (
        "Project code, experiment scripts and aggregate results",
        "https://github.com/al1naaa/clear_talk",
    ),
    (
        "FLEURS corpus and language subsets",
        "https://huggingface.co/datasets/google/fleurs",
    ),
    ("KSD official corpus record", "https://www.openslr.org/140/"),
    (
        "Golos far-field distribution",
        "https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield",
    ),
    ("LibriSpeech official corpus record", "https://www.openslr.org/12/"),
    (
        "Libri2Mix SepFormer model card",
        "https://huggingface.co/speechbrain/sepformer-libri2mix",
    ),
]:
    para(f'<link href="{url}" color="#006c66"><u>{label}</u></link>', 9.5, gap=3)
finish()
paper.save()
