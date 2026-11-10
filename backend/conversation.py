"""Conversation mode: long recordings, several speakers, Kazakh / Russian / English.

    speech detection ──► speaker turns ──► language of each turn ──► recognizer for that language ──► text

Everything runs on this computer from local files. The language is either fixed by the user or
detected turn by turn among the three supported languages, so a recording may change language
between turns. Each language goes to the recognizer the model profile names for it ("routes"),
or to one recognizer chosen by the user.
"""

import gc
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .background import Levels
from .config import LANGUAGES, Settings
from .loops import cut_loops
from .overlap import MAX_SECONDS as OVERLAP_MAX_SECONDS
from .overlap import MIN_SPEAKER_SECONDS, active_spans, band_limited, same_words, voice_print, who_is_who
from .multilingual import (LANGUAGE_NAMES, POLICY_ID, WRITES_WHAT_IT_HEARS, detect_language, languages_of_text,
                           transcribe_window)
from .speakers import MERGE_GAP_SECONDS, SAMPLE_RATE, SpeakerEncoder, Turn, diarize, joined_regions, speech_regions

logger = logging.getLogger(__name__)
PAD_SECONDS = 0.15             # audio kept around a turn so word edges are not cut
MIN_DETECT_SECONDS = 2.0       # shorter turns take their language from context
MAX_SPEAKER_CHOICE = 8
# Graphics memory. A recognizer needs about 1.25 times the size of its model file once loaded (measured on the
# GTX 1660 Ti: 2.0 GB for a 1.6 GB large model), and about 1 GB is left for CUDA itself, the speaker model and
# the desktop. Above the card's memory Windows does not fail: it spills into shared memory and everything
# crawls, so recognizers that do not fit are unloaded first.
LOADED_PER_FILE_BYTE = 1.25
RESERVED_GRAPHICS_BYTES = 1.0e9


def pick_device(preference: str) -> tuple[str, str]:
    """('cuda', 'int8_float32') when a working CUDA PyTorch is installed, else ('cpu', 'int8').

    Importing torch first also makes its bundled CUDA libraries loadable by CTranslate2 on Windows.
    int8 weights with float32 arithmetic gave the same transcripts as float32 on the GTX 1660 Ti and
    use less than half the memory, so two large recognizers fit on a 6 GB card.
    """
    if preference != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "int8_float32"
        except Exception:  # CPU-only installation
            logger.info("CUDA is not usable; conversation mode runs on the CPU")
        if preference == "cuda":
            raise RuntimeError("CLEARTALK_DEVICE=cuda, but no working CUDA device was found")
    return "cpu", "int8"


def settle_languages(turns, detections) -> list[str]:
    """Final language per turn: trusted detections stay, short or uncertain turns follow their context.

    A short turn takes the language its own speaker uses most; if that speaker has no trusted turn,
    it takes the language of the nearest trusted turn; with nothing to lean on, its own best guess.
    """
    trusted = [None if d is None or d["uncertain"] else d["language"] for d in detections]
    by_speaker = {}
    for turn, language in zip(turns, trusted):
        if language:
            by_speaker.setdefault(turn.speaker, Counter())[language] += turn.end - turn.start
    final = []
    for index, (turn, language) in enumerate(zip(turns, trusted)):
        if language is None and turn.speaker in by_speaker:
            language = by_speaker[turn.speaker].most_common(1)[0][0]
        if language is None:
            near = sorted((abs(other - index), value) for other, value in enumerate(trusted) if value)
            language = near[0][1] if near else (detections[index] or {}).get("language", "en")
        final.append(language)
    return final


class ConversationPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._recognizers = {}   # in order of use: the first one was used longest ago
        self._encoder = None
        self._device = None
        self._memory_limit = False  # False = not measured yet; None = no limit (CPU)
        self._separator = None

    def loaded_models(self) -> dict:
        return {"recognizers": sorted(self._recognizers), "speaker_labels": self._encoder is not None}

    def device(self) -> tuple[str, str]:
        if self._device is None:
            self._device = pick_device(self.settings.device)
        return self._device

    def memory_limit(self):
        """Bytes of graphics memory the recognizers may use together, or None when there is no such limit."""
        if self._memory_limit is False:
            self._memory_limit = None
            if self.device()[0] == "cuda":
                try:
                    import torch

                    self._memory_limit = torch.cuda.get_device_properties(0).total_memory - RESERVED_GRAPHICS_BYTES
                except Exception:  # the card cannot be asked: behave as before
                    logger.info("Graphics memory could not be read; recognizers are kept loaded")
        return self._memory_limit

    def footprint(self, key: str) -> float:
        """Estimated bytes a loaded recognizer takes."""
        weights = self.settings.multilingual_recognizer_dir(key) / "model.bin"
        return LOADED_PER_FILE_BYTE * weights.stat().st_size if weights.is_file() else 0.0

    def make_room(self, key: str) -> list[str]:
        """Unload the recognizers used longest ago until ``key`` fits in graphics memory; returns what was unloaded."""
        limit, unloaded = self.memory_limit(), []
        if limit is None:
            return unloaded
        while self._recognizers and sum(map(self.footprint, self._recognizers)) + self.footprint(key) > limit:
            oldest = next(iter(self._recognizers))
            model = self._recognizers.pop(oldest)
            try:
                model.model.unload_model()
            except Exception:  # an older CTranslate2: dropping the object frees the memory as well
                pass
            del model
            unloaded.append(oldest)
        if unloaded:
            gc.collect()
            logger.info("Unloaded %s to make room for %s", ", ".join(unloaded), key)
        return unloaded

    def recognizer(self, key: str):
        if key in self._recognizers:
            self._recognizers[key] = self._recognizers.pop(key)  # used now: last in line to be unloaded
            return self._recognizers[key]
        self.settings.verify_multilingual_recognizer(key)
        from faster_whisper import WhisperModel

        device, precision = self.device()
        self.make_room(key)
        self._recognizers[key] = WhisperModel(
            str(self.settings.multilingual_recognizer_dir(key)), device=device, compute_type=precision,
            cpu_threads=self.settings.cpu_threads, num_workers=1, local_files_only=True)
        return self._recognizers[key]

    def separator(self):
        """The two-voice separator for stretches where two people talk at once (backend/overlap.py)."""
        if self._separator is None:
            from .overlap import Separator

            self._separator = Separator(self.settings, self.device()[0])
        return self._separator

    def encoder(self):
        if self._encoder is None:
            self._encoder = SpeakerEncoder(self.settings.speaker_model_dir, self.device()[0], self.settings.cpu_threads)
        return self._encoder

    def route(self, model: str, language: str) -> str:
        """The recognizer for one language: the profile's route, or the single recognizer the user chose."""
        profile = self.settings.multilingual_profile
        routes = self.settings.multilingual_status()["routes"]  # routes among the installed recognizers
        if model == "auto":
            return routes[language]
        if language in profile["recognizers"][model]["languages"]:
            return model
        return routes[language]  # e.g. the Kazakh-only model met a Russian turn

    def mixed_route(self, model: str) -> str:
        """The recognizer for a recording whose languages change inside turns."""
        profile = self.settings.multilingual_profile
        if model != "auto" and set(profile["recognizers"][model]["languages"]) == set(LANGUAGES):
            return model
        return self.settings.multilingual_status()["mixed_recognizer"]

    def _separate_overlaps(self, audio, turns, speaker_turns, present, model, languages, items, used) -> set:
        """Separate the stretches of continuous speech in which the speaker labels show two people; where the
        separator finds two voices, add their sentences to ``items`` and return the indices of the turns they replace."""
        spans = []
        for start, end in speech_regions(audio):
            if len(present(start, end)) < 2:
                continue
            inside = {i for i, t in enumerate(turns) if min(end, t.end) > max(start, t.start)}
            low = min([start] + [turns[i].start for i in inside])
            high = max([end] + [turns[i].end for i in inside])
            if spans and low <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(high, spans[-1][1]), spans[-1][2] | inside)
            else:
                spans.append((low, high, inside))
        replaced, limited = set(), None
        writer = self.mixed_route(model)
        for low, high, inside in spans:
            if not inside or high - low > OVERLAP_MAX_SECONDS:
                continue
            offset = max(0.0, low - PAD_SECONDS)
            stretch = audio[int(offset * SAMPLE_RATE):int((high + PAD_SECONDS) * SAMPLE_RATE)]
            raw, loud, two_voices, _ = self.separator().split(stretch)
            if not two_voices:
                continue
            people = present(low, high)[:2]
            # Each track must sound like a different one of the two people (their voice prints over all their turns):
            # a voice spread over both tracks, or background talkers taken for a second voice, does not pass.
            if limited is None:
                limited = band_limited(audio)
            speaker_prints = [voice_print(self.encoder(), limited, [(t.start, t.end) for t in speaker_turns if t.speaker == person])
                              for person in people]
            mapping = who_is_who([voice_print(self.encoder(), raw[k], active_spans(raw[k])) for k in range(2)], speaker_prints)
            if mapping is None:
                continue
            who_of = {track: people[index] for track, index in mapping.items()}
            fallback = next((languages[i] for i in sorted(inside) if languages[i]), None) or "kk"
            pieces, heard_tracks = [], [transcribe_window(self.recognizer(writer), loud[track], None) for track in range(2)]
            if same_words(heard_tracks[0]["text"], heard_tracks[1]["text"]):
                continue  # one voice written twice: not two people; the turns are recognised as usual
            for track, heard in enumerate(heard_tracks):
                for segment in heard["segments"]:
                    # A sentence on a track that is much quieter there than the other track is the other voice
                    # leaking through the separator; writing it down would invent words.
                    if self.separator().own_voice(raw, track, segment["start"], segment["end"]):
                        pieces.append((offset + segment["start"], offset + segment["end"], who_of[track], segment["text"]))
            if not pieces:
                continue  # nothing usable on either track: the turns are recognised as usual
            pieces.sort(key=lambda p: p[0])
            used[writer] += 1
            replaced |= inside
            for number, (start, end, who, text) in enumerate(pieces):
                written = languages_of_text(text, fallback)
                items.append({"order": (min(inside), number), "start": round(start, 2), "end": round(end, 2), "speaker": f"Speaker {who + 1}",
                              "language": written[0] if len(written) == 1 else "mixed", "languages": written,
                              "language_name": " + ".join(LANGUAGE_NAMES[code] for code in written), "text": text,
                              "recognizer": writer, "needs_review": True, "overlap": True})
        return replaced

    def transcribe_two_voices(self, audio, language="auto", model="auto", update=None) -> dict:
        """The user says two people talk at the same time in this short recording: the two-voice separator splits the
        whole recording, and each voice is transcribed on its own (language detected turn by turn) as one speaker.

        Not measured on a test set. It is the separation that Conversation mode measured on validation recordings with
        overlap (round-4 log: errors 33.7% -> 16.0%) and left off by default, because on recordings WITHOUT two voices
        at once it wrote one voice twice or background talkers as a second voice. Here it runs only when chosen; when
        both tracks carry the same words, the recording is transcribed as usual."""
        update = update or (lambda stage, progress: None)
        audio = np.asarray(audio, dtype=np.float32)
        update("Separating two voices", 10)
        raw, loud, _, _ = self.separator().split(audio)
        if (len(raw) != 2 or len(loud) != 2 or
                any(np.asarray(track).shape != audio.shape or not np.isfinite(track).all() for track in [*raw, *loud])):
            raise RuntimeError("The separator did not return two finite tracks matching the input duration")
        voices = []
        for index in range(2):
            update(f"Transcribing voice {index + 1} of 2", 30 + 30 * index)
            voices.append(self.transcribe(loud[index], language, model, 1))
        texts = [" ".join(turn["text"] for turn in voice["turns"]) for voice in voices]
        if not all(texts) or same_words(*texts):
            update("Could not confirm two distinct transcripts: using the original recording", 70)
            result = self.transcribe(audio, language, model, 2)
            result["two_voices"] = False
            result["separation_status"] = "fallback_original"
            return result
        turns, background, used = [], [], Counter()
        # The voice that starts talking first is Speaker 1.
        order = sorted(range(2), key=lambda index: min((turn["start"] for turn in voices[index]["turns"]), default=0.0))
        for number, index in enumerate(order, start=1):
            voice = voices[index]
            for turn in voice["turns"]:
                # A sentence on a track that is much quieter there than the other track is the other voice leaking
                # through the separator.
                if self.separator().own_voice(raw, index, turn["start"], turn["end"]):
                    turns.append({**turn, "speaker": f"Speaker {number}", "overlap": True})
            background += [{**turn, "speaker": f"Speaker {number}"} for turn in voice["background_turns"]]
            used.update(voice["recognizers"])
        turns.sort(key=lambda turn: (turn["start"], turn["speaker"]))
        return {"turns": turns, "speech_seconds": round(sum(voice["speech_seconds"] for voice in voices), 2),
                "recognizers": dict(used), "speaker_labels": True, "background_turns": background,
                "separated_overlaps": True, "two_voices": True,
                "separation_status": "two_tracks_returned", "speaker_label_method": "separated_source_tracks",
                # Internal waveform payload only: process() writes it to private job files and
                # returns JSON metadata, never numpy arrays. Preserve the same order as the labels.
                "_source_audio": [loud[index] for index in order]}

    def transcribe(self, audio, language="auto", model="auto", speakers=None, update=None, skip_detection=None,
                   uncertain_to_mixed=None, cross_check=None, by_region=None, separate_overlaps=None,
                   hide_background=None, cut_repeats=True, text_first=None) -> dict:
        """Turns with speaker, language and text for mono 16 kHz audio.

        ``language`` is a language code, "auto" (one language per turn, detected) or "mixed" (languages may
        change inside a turn: every turn goes to the recognizer trained for that, and the turn's languages
        are read from the text it wrote).

        Two decoding options, read from the model profile unless given (both off in the validated profile):
        ``skip_detection`` - with "mixed", the language detector is not run (the recognizer chooses its token);
        ``uncertain_to_mixed`` - with "auto", a turn whose language detection is uncertain goes to the recognizer
        of the mixed setting instead of its language's route;
        ``cross_check`` - with "auto", a turn routed to a recognizer that writes in the language it is told (large-v3-turbo
        for English) is also written by the recognizer of the mixed setting; if that one heard Kazakh or Russian, its text
        is kept (a confident detector can still call a Kazakh phrase English, and the turn would come out as English);
        ``by_region`` - the recognizer gets every stretch of continuous speech whole, and its sentences are given to the
        speakers afterwards by time. Cutting speech at speaker-label changes can split one person's sentence; the
        recognizer then misses or guesses the words at the cut;
        ``separate_overlaps`` - a stretch of continuous speech in which the speaker labels show two people is split by
        the two-voice separator; if both voices carry speech, each is recognised on its own by the recognizer of the
        mixed setting and given to one of the two speakers (flagged for checking), in place of the turns of that
        stretch; everything else is transcribed as without it;
        ``hide_background`` - a short turn that is much quieter than the conversation and barely above the recording's
        background level (backend/background.py) is taken out of the transcript and returned in "background_turns";
        ``cut_repeats`` - repetition loops are cut back (backend/loops.py); off only to measure the app as it was before;
        ``text_first`` - with "auto", every turn is written first by the recognizer of the mixed setting (it writes what it
        hears, in any of the three languages); a turn it wrote in English is written again by the English route. The
        language detector is not run: its confident mistakes (a Kazakh phrase taken for English) cannot happen.
        """
        profile_options = self.settings.multilingual_profile.get("options", {})
        skip_detection = profile_options.get("mixed_skip_detection", False) if skip_detection is None else skip_detection
        uncertain_to_mixed = profile_options.get("uncertain_to_mixed", False) if uncertain_to_mixed is None else uncertain_to_mixed
        cross_check = profile_options.get("cross_check", False) if cross_check is None else cross_check
        by_region = profile_options.get("transcribe_by_region", False) if by_region is None else by_region
        separate_overlaps = profile_options.get("separate_overlaps", False) if separate_overlaps is None else separate_overlaps
        separate_overlaps = bool(separate_overlaps and self.settings.separator_dir.is_dir())
        hide_background = profile_options.get("hide_background_turns", False) if hide_background is None else hide_background
        text_first = profile_options.get("route_by_text", False) if text_first is None else text_first
        text_first = bool(text_first and language == "auto" and self.mixed_route(model) in WRITES_WHAT_IT_HEARS)
        skip_detection = bool(skip_detection and language == "mixed" and self.mixed_route(model) in WRITES_WHAT_IT_HEARS)
        update = update or (lambda stage, progress: None)
        audio = np.asarray(audio, dtype=np.float32)
        profile = self.settings.multilingual_profile
        update("Finding speech and speaker turns", 12)
        labels = self.settings.multilingual_status()["speaker_labels"] and speakers != 1
        speaker_turns = diarize(audio, self.encoder() if labels else None, speakers)
        turns = speaker_turns

        def present(start, end):
            """Speakers talking at least MIN_SPEAKER_SECONDS in [start, end], longest first."""
            seconds = Counter()
            for t in speaker_turns:
                if min(end, t.end) > max(start, t.start):
                    seconds[t.speaker] += min(end, t.end) - max(start, t.start)
            return [speaker for speaker, talked in seconds.most_common() if talked >= MIN_SPEAKER_SECONDS]

        def speaker_at(start, end):
            """The speaker who talks longest in [start, end], or the nearest one."""
            overlaps = [(min(end, t.end) - max(start, t.start), t.speaker) for t in speaker_turns]
            best = max(overlaps, default=(0.0, 0))
            if best[0] > 0:
                return best[1]
            return min(speaker_turns, key=lambda t: min(abs(t.start - end), abs(t.end - start))).speaker if speaker_turns else 0

        if by_region and speaker_turns:  # recognise whole stretches of speech; speakers are attached afterwards
            # A pause of a breath is closed only when the same person talks on both sides of it.
            same = lambda a, b: speaker_at(*a) == speaker_at(*b)  # noqa: E731
            turns = [Turn(start, end, speaker_at(start, end)) for start, end in joined_regions(speech_regions(audio), same=same)]

        def piece(turn):
            start = max(0, int((turn.start - PAD_SECONDS) * SAMPLE_RATE))
            return audio[start:int((turn.end + PAD_SECONDS) * SAMPLE_RATE)]

        if skip_detection or text_first:
            detections, languages = [None] * len(turns), [None] * len(turns)
        elif language in ("auto", "mixed"):
            update("Detecting the language of each turn", 25)
            detector = self.recognizer(self.settings.multilingual_status()["detector"])
            detections = [detect_language(detector, piece(turn)) if turn.end - turn.start >= MIN_DETECT_SECONDS
                          else None for turn in turns]
            if turns and not any(detections):  # only short turns: decide once from all speech together
                joined = np.concatenate([piece(turn) for turn in turns])[:30 * SAMPLE_RATE]
                detections[0] = {**detect_language(detector, joined), "uncertain": False}
            languages = settle_languages(turns, detections)
        else:
            detections = [None] * len(turns)
            languages = [language] * len(turns)

        # Turns are independent of each other, so they are transcribed recognizer by recognizer (those already
        # loaded first) and put back in spoken order: a small graphics card then holds one recognizer at a time
        # instead of loading and unloading them at every change of language.
        def key_of(turn_language, detection):
            if language == "mixed" or text_first:
                return self.mixed_route(model)
            if uncertain_to_mixed and language == "auto" and (detection is None or detection["uncertain"]):
                candidate = self.mixed_route(model)
                if candidate in WRITES_WHAT_IT_HEARS:
                    return candidate
            return self.route(model, turn_language)

        keys = [key_of(turn_language, detection) for turn_language, detection in zip(languages, detections)]
        items, used, replaced = [], Counter(), set()
        if separate_overlaps and speaker_turns:
            update("Separating people who talk at the same time", 28)
            replaced = self._separate_overlaps(audio, turns, speaker_turns, present, model, languages, items, used)
        first_use = {key: index for index, key in reversed(list(enumerate(keys)))}
        order = sorted(range(len(turns)), key=lambda index: (keys[index] not in self._recognizers, first_use[keys[index]], index))
        for done, index in enumerate(order):
            turn, turn_language, key = turns[index], languages[index], keys[index]
            update(f"Transcribing turn {done + 1} of {len(turns)}", 30 + int(65 * done / max(len(turns), 1)))
            if index in replaced:  # written from the separated voices
                continue
            result = transcribe_window(self.recognizer(key), piece(turn), turn_language)
            if cross_check and language == "auto" and key not in WRITES_WHAT_IT_HEARS:
                checker = self.mixed_route(model)
                if checker in WRITES_WHAT_IT_HEARS:
                    second = transcribe_window(self.recognizer(checker), piece(turn), None)
                    heard = languages_of_text(second["text"], turn_language) if second["text"] else [turn_language]
                    # cross_check "kazakh": only a turn in which our model wrote Kazakh (Kazakh-only letters) is taken over
                    wanted = ("kk",) if cross_check == "kazakh" else tuple(code for code in LANGUAGES if code != turn_language)
                    if second["text"] and turn_language not in heard and any(code in heard for code in wanted):
                        result, key = second, checker
            if text_first and result["text"] and languages_of_text(result["text"], "kk") == ["en"]:
                english = self.route(model, "en")
                if english != key:  # our model heard English: the English route writes it better
                    again = transcribe_window(self.recognizer(english), piece(turn), "en")
                    if again["text"]:
                        result, key, turn_language = again, english, "en"
            if not result["text"]:
                continue
            turn_language = turn_language or result["language"] or "kk"
            used[key] += 1
            detection = detections[index]
            doubtful = language == "auto" and not text_first and (detection is None or detection["uncertain"])
            if by_region and speaker_turns:
                # Give every sentence of the stretch to the speaker who was talking then.
                offset = max(0.0, turn.start - PAD_SECONDS)
                groups = []
                for segment in result["segments"]:
                    start, end = offset + segment["start"], offset + segment["end"]
                    who = speaker_at(start, end)
                    if groups and groups[-1]["speaker"] == who:
                        groups[-1].update(end=end, text=groups[-1]["text"] + " " + segment["text"],
                                          review=groups[-1]["review"] or segment["needs_review"])
                    else:
                        groups.append({"start": start, "end": end, "speaker": who, "text": segment["text"], "review": segment["needs_review"]})
                pieces = [(g["start"], g["end"], g["speaker"], g["text"], g["review"]) for g in groups]
            else:
                pieces = [(turn.start, turn.end, turn.speaker, result["text"],
                           any(segment["needs_review"] for segment in result["segments"]))]
            for number, (start, end, who, text, review) in enumerate(pieces):
                piece_language, written = turn_language, [turn_language]
                if key in WRITES_WHAT_IT_HEARS and language in ("auto", "mixed"):
                    # Its text names the language better than its token, and one turn may hold two languages.
                    written = languages_of_text(text, turn_language)
                    piece_language = written[0] if len(written) == 1 else "mixed"
                items.append({
                    "order": (index, number),
                    "start": round(start, 2), "end": round(end, 2), "speaker": f"Speaker {who + 1}",
                    "language": piece_language, "languages": written,
                    "language_name": " + ".join(LANGUAGE_NAMES[code] for code in written), "text": text,
                    "recognizer": key, "needs_review": bool(review or doubtful),
                })
        items.sort(key=lambda item: item.pop("order"))
        for item in items if cut_repeats else []:  # a recogniser stuck in a loop writes one phrase dozens of times
            item["text"], looped = cut_loops(item["text"])
            item["needs_review"] = bool(item["needs_review"] or looped)
        background = []
        if hide_background and items:
            levels = Levels(audio)
            quiet = [not item.get("overlap") and levels.is_background(item["start"], item["end"]) for item in items]
            background = [item for item, flag in zip(items, quiet) if flag]
            items = [item for item, flag in zip(items, quiet) if not flag]
        if by_region and speaker_turns:  # one speaker's sentences in a row form one turn
            merged = []
            for item in items:
                last = merged[-1] if merged else None
                if (last and last["speaker"] == item["speaker"] and item["start"] - last["end"] <= MERGE_GAP_SECONDS
                        and last["recognizer"] == item["recognizer"] and last.get("overlap") == item.get("overlap")):
                    last.update(end=item["end"], text=last["text"] + " " + item["text"], needs_review=last["needs_review"] or item["needs_review"])
                    if item["language"] != last["language"]:
                        codes = list(dict.fromkeys(last["languages"] + item["languages"]))
                        last.update(languages=codes, language="mixed" if len(codes) > 1 else codes[0],
                                    language_name=" + ".join(LANGUAGE_NAMES[code] for code in codes))
                else:
                    merged.append(dict(item))
            items = merged
        # Speaker labels are numbered 1, 2, 3 … in the order people first speak in the transcript (labels that only
        # background stretches had come after them), so no number is skipped.
        numbers = {}
        for item in items + background:
            numbers.setdefault(item["speaker"], f"Speaker {len(numbers) + 1}")
        for item in items + background:
            item["speaker"] = numbers[item["speaker"]]
        return {"turns": items, "speech_seconds": round(sum(turn.end - turn.start for turn in turns), 2),
                "recognizers": dict(used), "speaker_labels": bool(labels), "background_turns": background,
                "separated_overlaps": separate_overlaps}

    def process(self, job_id: str, job_dir: Path, context: dict, update) -> dict:
        import soundfile as sf

        started = time.perf_counter()
        language = context.get("language", "auto")
        model = context.get("model", "auto")
        speakers = context.get("speakers") or None
        profile = self.settings.multilingual_profile
        if language not in ("auto", "mixed") and language not in LANGUAGES:
            raise ValueError("Unsupported language")
        if model != "auto" and model not in profile["recognizers"]:
            raise ValueError("Unsupported speech recognizer")
        audio, sample_rate = sf.read(job_dir / "original.wav", dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise ValueError("The stored recording must be mono 16 kHz audio")
        duration = len(audio) / sample_rate
        update("Loading the speech models", 5)
        if context.get("two_voices"):
            result = self.transcribe_two_voices(audio, language, model, update)
        else:
            result = self.transcribe(audio, language, model, speakers, update)
        turns = result["turns"]
        sources = []
        for index, source in enumerate(result.pop("_source_audio", [])):
            if index >= 2 or np.asarray(source).shape != audio.shape or not np.isfinite(source).all():
                raise RuntimeError("Invalid separated audio cannot be saved")
            label = f"Speaker {index + 1}"
            filename = "source-a.wav" if index == 0 else "source-b.wav"
            sf.write(job_dir / filename, source, SAMPLE_RATE, subtype="PCM_16")
            source_turns = [turn for turn in turns if turn["speaker"] == label]
            sources.append({
                "label": label, "speaker": label,
                "audio_url": f"/media/{job_id}/{filename}",
                "text": " ".join(turn["text"] for turn in source_turns),
                "segments": [{"start": turn["start"], "end": turn["end"], "text": turn["text"],
                              "needs_review": True} for turn in source_turns],
                "review_notes": ["Estimated separated source, not an identified person. Playback levels are normalized independently; words and other voices may remain."],
            })
        actual_two_voices = bool(result.get("two_voices") and len(sources) == 2)
        label_method = ("separated_source_tracks" if actual_two_voices else
                        "ecapa_clustering" if result["speaker_labels"] else "single_speaker")

        speaker_summary, language_seconds = {}, Counter()
        for turn in turns:
            entry = speaker_summary.setdefault(turn["speaker"], {"label": turn["speaker"], "seconds": 0.0, "turns": 0, "languages": []})
            entry["seconds"] = round(entry["seconds"] + turn["end"] - turn["start"], 2)
            entry["turns"] += 1
            for code in turn["languages"]:
                if LANGUAGE_NAMES[code] not in entry["languages"]:
                    entry["languages"].append(LANGUAGE_NAMES[code])
            language_seconds[turn["language_name"]] += turn["end"] - turn["start"]

        device, precision = self.device()
        names = [profile["recognizers"][key]["name"] for key in result["recognizers"]]
        warnings = [
            "Machine-generated transcript in Kazakh, Russian or English; it can contain errors. Check the audio before relying on it.",
            "Turn times are approximate. Review flags use simple score thresholds, not calibrated confidence.",
        ]
        if context.get("two_voices"):
            warnings.append("Two people at the same time: the whole recording was split into two voices by the separator of the "
                            "English two-voice mode, and each voice was transcribed on its own. This setting is a demonstration, "
                            "not measured on a test set; words can be lost or given to the wrong voice."
                            if actual_two_voices else
                            "Two people at the same time was chosen, but two distinct usable transcripts could not be "
                            "confirmed. The original recording was transcribed instead. This does not prove only one "
                            "person spoke, and overlapping words may be missing.")
        elif result.get("separated_overlaps"):
            warnings.append("Where two people spoke at the same time, their voices were separated and written as two turns "
                            "marked for checking; words can still be lost or given to the wrong speaker.")
        else:
            warnings.append("People speaking at the same time are not separated in this mode: overlapping words can be lost "
                            "or attributed to one speaker.")
        hidden = result.get("background_turns", [])
        if hidden:
            warnings.append(f"Short quiet stretches left out of the transcript as background voices or noise: {len(hidden)}. "
                            "They are listed below it.")
        if result["speaker_labels"]:
            warnings.append("Speaker 1, Speaker 2, … are labels inside this recording, not identified people. "
                            "Short interjections can be attached to the wrong speaker.")
        else:
            warnings.append("Speaker labels were not computed for this recording: all text is shown as one speaker.")
        if language == "auto":
            warnings.append("The language was detected turn by turn among Kazakh, Russian and English. A language change "
                            "in the middle of a turn, or a very short turn, can be written in the wrong language.")
        if language == "mixed":
            warnings.append("Mixed-language setting: every turn was written by the recognizer trained on recordings that change "
                            "language. It was trained and measured on assembled recordings, not on one person switching language "
                            "inside a sentence; check such places against the audio.")
        if any(key in WRITES_WHAT_IT_HEARS for key in result["recognizers"]):
            warnings.append("The ClearTalk multilingual recognizer was fine-tuned on read sentences and short phrases; "
                            "its measured gains do not guarantee better text on every recording.")
        update("Preparing the transcript", 97)
        return {
            "duration_seconds": round(duration, 2),
            "processing_seconds": round(time.perf_counter() - started, 2),
            "mode": "conversation",
            "model": model,
            "language": language,
            "two_voices": actual_two_voices,
            "requested_two_voices": bool(context.get("two_voices")),
            "separation_status": result.get("separation_status", "not_requested"),
            "speaker_label_method": label_method,
            "original": {"audio_url": f"/media/{job_id}/original.wav", "text": " ".join(turn["text"] for turn in turns),
                         "segments": [{"start": t["start"], "end": t["end"], "text": t["text"], "needs_review": t["needs_review"]}
                                      for t in turns],
                         "review_notes": [] if turns else ["No speech was transcribed. Quiet or heavily distorted speech can be missed."]},
            "sources": sources,
            "turns": turns,
            "background_turns": [{"start": t["start"], "end": t["end"], "speaker": t["speaker"], "text": t["text"]} for t in hidden],
            "speakers": list(speaker_summary.values()),
            "languages": [{"name": name, "seconds": round(seconds, 2)} for name, seconds in language_seconds.most_common()],
            "speech_seconds": result["speech_seconds"],
            "metrics": [],
            "warnings": warnings,
            "models": {"separator": f"{self.settings.separator_id} (8 kHz, resampled to 16 kHz for playback)" if actual_two_voices else None,
                       "asr": f"{' + '.join(names) or 'no recognizer used'} · {device.upper()} {precision} · speech detection",
                       "speakers": profile["speaker_model"]["id"] if label_method == "ecapa_clustering" else None,
                       "asr_policy": POLICY_ID},
            "download_url": f"/api/jobs/{job_id}/download",
        }
