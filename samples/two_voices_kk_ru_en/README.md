# Two people at the same time - Kazakh, Russian, English

Three short demonstration clips in which a second person starts talking while the first is still speaking
(3.7-4.8 s of overlap). Built on 2026-10-19 from the sentences of the packaged three-language demo
(`data/demo/conversation.wav`, FLEURS training part, CC BY 4.0): two sentences by different readers are levelled
and added together. `manifest.json` lists each clip's two sentences, languages and start times.

| File | First voice | Second voice (starts at) |
|---|---|---|
| `kazakh_and_russian.wav` | Kazakh, woman | Russian, man (3.0 s) |
| `russian_and_kazakh.wav` | Russian, woman | Kazakh, man (2.5 s) |
| `english_and_kazakh.wav` | English, woman | Kazakh, man (2.0 s) |

In the app: Conversation mode → Speakers: "2 · at the same time" → choose a clip → Transcribe conversation.
Without that setting Conversation mode writes only one of the two voices.

These are assembled clips of read sentences, not a measurement: the two-voice setting was not scored on a test
set. Its separator was measured inside Conversation mode on validation recordings with overlap (round-4 log:
errors 33.7% → 16.0%) and is left off by default because on recordings without two voices at once it wrote
extra text.
