/* ClearTalk interface languages.

   The page and app.js are written in English. This layer shows the same interface in Russian by
   translating text and a few attributes as they appear on the page, and can switch back at any time.
   Recognized speech is never touched: transcripts, speaker names and file names are left as they are.

   Every English sentence of the interface needs an entry here (a fixed sentence in EXACT, a sentence
   with numbers or names in PATTERNS); tests/test_interface_language.py fails when one is missing. */
(() => {
  'use strict';

  const LANGUAGES = { Kazakh: 'казахский', Russian: 'русский', English: 'английский', Mixed: 'смешанный' };
  const MODELS = {
    'Kazakh Whisper large-v3-turbo': 'Казахская Whisper large-v3-turbo',
    'Whisper small (multilingual)': 'Whisper small (многоязычная)',
    'ClearTalk large multilingual (LoRA fine-tuned)': 'ClearTalk большая многоязычная (дообучена, LoRA)',
    'ClearTalk multilingual (LoRA fine-tuned)': 'ClearTalk многоязычная (дообучена, LoRA)',
    'ClearTalk multilingual with Kazakh broadcast (LoRA fine-tuned)': 'ClearTalk многоязычная с казахским эфиром (дообучена, LoRA)',
    'ClearTalk small (Kazakh broadcast)': 'ClearTalk малая (казахский эфир)',
    'Pretrained Whisper small.en': 'Готовая Whisper small.en',
    'ClearTalk large': 'ClearTalk большая',
    'ClearTalk small': 'ClearTalk малая',
    'no recognizer used': 'модель распознавания не понадобилась',
  };
  const SETS = {
    'mixed-language recordings': 'записи со сменой языка',
    'Kazakh read sentences': 'казахские прочитанные предложения',
    'Russian read sentences': 'русские прочитанные предложения',
    'English read sentences': 'английские прочитанные предложения',
    'Kazakh broadcast and read speech': 'казахская эфирная и прочитанная речь',
    'short Russian phrases': 'короткие русские фразы',
    'two-speaker mixtures': 'смеси двух говорящих',
    'single clean speaker': 'один говорящий без шума',
  };

  const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);
  const language = (name) => LANGUAGES[name] ? capital(LANGUAGES[name]) : name;
  const languages = (text, separator) => text.split(separator).map(name => language(name.trim())).join(separator);
  const decimal = (text) => String(text).replace(/(\d)\.(\d)/g, '$1,$2');
  const thousands = (text) => String(text).replace(/,/g, ' ');
  const duration = (text) => /^\d+\.\ds$/.test(text) ? `${decimal(text.slice(0, -1))} с` : text;
  const speaker = (name) => name.replace(/^Speaker (\d+)$/, 'Говорящий $1');
  const source = (name) => name.replace(/^Source ([A-Z])$/, 'Источник $1').replace(/^original recording$/i, 'исходной записи');
  const escaped = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const modelNames = new RegExp(Object.keys(MODELS).sort((a, b) => b.length - a.length).map(escaped).join('|'), 'g');
  const models = (text) => text.replace(modelNames, name => MODELS[name]);
  const system = (name) => models(name).replace(/^pretrained$/, 'готовая');
  const plural = (count, one, few, many) => {
    const tail = Math.abs(Number(count)) % 100, last = tail % 10;
    return tail > 10 && tail < 20 ? many : last === 1 ? one : last >= 2 && last <= 4 ? few : many;
  };
  const NO_LABELS = ' Метки говорящих не установлены: весь текст показан как один говорящий.';

  const EXACT = {
    'Kazakh + Russian: overlapping voices': 'Казахский + русский: два голоса одновременно',
    'Controlled demonstration, not an accuracy benchmark.': 'Составленный пример, не бенчмарк точности.',
    'Could not confirm two distinct transcripts: using the original recording': 'Не удалось подтвердить два разных текста: используем исходную запись',
    'Two people at the same time was chosen, but two distinct usable transcripts could not be confirmed. The original recording was transcribed instead. This does not prove only one person spoke, and overlapping words may be missing.': 'Выбраны два одновременных голоса, но два разных пригодных текста получить не удалось. Распознана исходная запись. Это не доказывает, что говорил один человек: наложенные слова могут отсутствовать.',
    'Estimated separated source, not an identified person. Playback levels are normalized independently; words and other voices may remain.': 'Выделенная дорожка, не установленная личность. Громкость дорожек нормализована отдельно; возможны пропуски слов и остатки другого голоса.',
    // ---- page frame
    'ClearTalk - Audio workspace': 'ClearTalk - работа со звуком',
    'Skip to workspace': 'Перейти к рабочей области',
    'Project navigation': 'Навигация по проекту',
    'ClearTalk home': 'ClearTalk, главная',
    'YOUR WORKSPACE': 'РАБОЧАЯ ОБЛАСТЬ',
    'Audio workbench': 'Работа со звуком',
    'Project & methods': 'О проекте и методах',
    'Make room': 'Место',
    'for every voice.': 'для каждого голоса.',
    'Speech separation meets': 'Разделение речи',
    'natural language processing.': 'и обработка языка вместе.',
    'RESEARCH PROTOTYPE': 'ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП',
    'Checking engine': 'Проверяем модели',
    'Speech engine ready': 'Модели готовы',
    'Model setup needed': 'Модели не установлены',
    'Engine unavailable': 'Программа не отвечает',
    'Workspace locked': 'Нужен ключ доступа',
    'KAZAKH · RUSSIAN · ENGLISH': 'КАЗАХСКИЙ · РУССКИЙ · АНГЛИЙСКИЙ',
    'Interface language': 'Язык интерфейса',
    'SIGNAL THROUGH THE NOISE': 'СИГНАЛ СКВОЗЬ ШУМ',
    'One recording.': 'Одна запись.',
    'Every voice, clearer.': 'Каждый голос - понятнее.',
    'Transcribe a conversation in Kazakh, Russian or English and see who spoke when,': 'Расшифруйте разговор на казахском, русском или английском и увидьте, кто и когда говорил,',
    'or separate two voices that talk over each other.': 'или разделите два голоса, которые звучат одновременно.',
    'MIXTURE → INDIVIDUAL SOURCES': 'СМЕСЬ → ОТДЕЛЬНЫЕ ГОЛОСА',
    '01 / SPEECH': '01 / РЕЧЬ',
    '02 / CLARITY': '02 / ЯСНОСТЬ',
    'This workspace is protected': 'Рабочая область защищена',
    'Enter the access key provided by the workspace owner.': 'Введите ключ доступа, который дал владелец.',
    'Workspace access key': 'Ключ доступа',
    'Unlock workspace': 'Открыть',

    // ---- input panel
    'Start with audio': 'Начните с записи',
    'INPUT': 'ВХОД',
    'Choose an audio file': 'Выберите аудиофайл',
    'Drop a recording here': 'Перетащите запись сюда',
    'or': 'или',
    'browse files': 'выберите файл',
    'WAV, MP3, M4A & more': 'WAV, MP3, M4A и другие',
    'Record from the microphone': 'Записать с микрофона',
    'Remove selected audio': 'Убрать выбранную запись',
    'OR TRY IT FIRST': 'ИЛИ СНАЧАЛА ПОПРОБУЙТЕ',
    'Two voices. One recording.': 'Два голоса. Одна запись.',
    'Three languages. Six voices.': 'Три языка. Шесть голосов.',
    'Load the included overlapping-speech demo': 'Загрузить готовый пример с наложением голосов',
    'Kazakh, Russian and English, one after another · assembled demo': 'Казахский, русский и английский друг за другом · собранный пример',
    'Two audiobook voices + noise · controlled demo': 'Два голоса из аудиокниг и шум · учебный пример',
    'Demo unavailable for this mode - upload your own recording': 'Для этого режима примера нет - загрузите свою запись',
    'Included demo': 'Готовый пример',
    'Two voices, one noisy recording': 'Два голоса в одной шумной записи',
    'Two voices in noise': 'Два голоса в шуме',
    'Three languages, six readers': 'Три языка, шесть дикторов',
    'A short, reproducible two-speaker demonstration.': 'Короткий воспроизводимый пример с двумя говорящими.',
    'Two public LibriSpeech audiobook excerpts mixed with generated white noise at 10 dB SNR. A controlled demonstration, not a real meeting or benchmark result.': 'Два отрывка из открытых аудиокниг LibriSpeech, смешанные с белым шумом (сигнал громче шума на 10 дБ). Это учебный пример, а не настоящая встреча и не результат испытаний.',
    'Six read sentences in Kazakh, Russian and English placed one after another. An assembled demonstration, not a natural conversation.': 'Шесть прочитанных предложений на казахском, русском и английском, поставленных друг за другом. Это собранный пример, а не живой разговор.',
    'Demo loaded · three languages, six readers': 'Пример загружен · три языка, шесть дикторов',
    'Demo loaded · two overlapping voices': 'Пример загружен · два голоса одновременно',
    'Processing mode': 'Режим обработки',
    'CHOOSE YOUR WORKFLOW': 'ВЫБЕРИТЕ, ЧТО НУЖНО',
    'Conversation': 'Разговор',
    'Several speakers taking turns · Kazakh, Russian, English · long recordings': 'Несколько человек говорят по очереди · казахский, русский, английский · длинные записи',
    'Separate + transcribe': 'Разделить и расшифровать',
    'Two voices at once · English · short clip': 'Два голоса одновременно · английский · короткий фрагмент',
    'Transcribe only': 'Только расшифровать',
    'One voice · English · short clip': 'Один голос · английский · короткий фрагмент',
    'Language': 'Язык',
    'Detect automatically': 'Определить автоматически',
    'Languages change inside turns': 'Языки меняются в реплике',
    'Kazakh · Қазақша': 'Казахский · Қазақша',
    'Russian · Русский': 'Русский',
    'English': 'Английский',
    'Kazakh': 'Казахский',
    'Russian': 'Русский',
    'Mixed': 'Смешанный',
    'Speakers': 'Говорящие',
    'Find automatically': 'Автоматически',
    '1 · no speaker labels': '1 · без меток говорящих',
    '2 · at the same time': '2 · говорят одновременно',
    'Speech recognizer': 'Модель распознавания',
    'Best measured model for each language': 'Лучшая по замерам модель для каждого языка',
    'Each turn is written by the recognizer that scored best for its language on validation data.': 'Каждую реплику записывает модель, которая показала лучший результат для этого языка на проверочных данных.',
    'About our multilingual model': 'О нашей многоязычной модели',
    'About our multilingual models': 'О наших многоязычных моделях',
    'Pretrained Whisper small.en': 'Готовая Whisper small.en',
    'ClearTalk fine-tuned (ours)': 'ClearTalk, дообученная (наша)',
    'Whisper small.en as released by OpenAI, without ClearTalk training.': 'Whisper small.en в том виде, как её выпустила OpenAI, без дообучения в ClearTalk.',
    'Whisper small.en adapted in this project (LoRA) to separated, noisy speech. Compare it with the pretrained model on your own recordings.': 'Whisper small.en, дообученная в этом проекте (LoRA) на разделённой шумной речи. Сравните её с готовой моделью на своих записях.',
    'About our trained model': 'О нашей обученной модели',
    'Process recording': 'Обработать запись',
    'Processing your recording…': 'Обрабатываем запись…',
    'Transcribe conversation': 'Расшифровать разговор',
    'Transcribe recording': 'Расшифровать запись',
    'Select a file or load the demo to begin.': 'Выберите файл или загрузите пример, чтобы начать.',
    'Your recording is being processed by the speech models.': 'Запись обрабатывают модели распознавания.',
    'Enter the workspace access key above.': 'Введите ключ доступа выше.',
    'The models for this mode are not installed. Choose another mode or complete setup.': 'Модели для этого режима не установлены. Выберите другой режим или завершите установку.',
    'Processed in this workspace. Results are private to this browser session.': 'Обработка идёт в этой рабочей области. Результаты видны только в этом браузере.',
    'Designed for': 'Рассчитано на',
    'two overlapping speakers': 'двух человек, говорящих одновременно',
    '. Source labels apply to this clip. Listen back to check the words.': '. Метки голосов относятся только к этому фрагменту. Сверяйте слова с записью.',
    'For people who take turns. Voices that talk at the same time are separated only with Speakers: "2 · at the same time" (clips up to 30 s, a demonstration), and speaker labels are not identities. Listen back to check the words.': 'Для записей, где говорят по очереди. Одновременную речь этот режим разделяет, только если в «Говорящие» выбрать «2 · говорят одновременно» (записи до 30 с, демонстрация), а метки говорящих - не установленные личности. Сверяйте слова с записью.',
    'Designed for two overlapping speakers, English only. Source labels apply to this clip. Listen back to check the words.': 'Рассчитано на двух человек, говорящих одновременно, только английский. Метки голосов относятся только к этому фрагменту. Сверяйте слова с записью.',
    'One English recording, no separation and no speaker labels. Listen back to check the words.': 'Одна английская запись, без разделения и без меток говорящих. Сверяйте слова с записью.',
    'Check this recording': 'Проверить запись',
    'Delete retained recording': 'Удалить сохранённую запись',
    'This recording is still retained privately for your browser. You can delete its audio now.': 'Эта запись ещё хранится и видна только вашему браузеру. Её можно удалить сейчас.',
    'This recording may still be processing. Its link is saved in this page address. Check again to recover the result; active recordings cannot be deleted.': 'Возможно, запись ещё обрабатывается. Ссылка на неё сохранена в адресе страницы. Проверьте ещё раз, чтобы получить результат; запись в обработке удалить нельзя.',

    // ---- results panel
    'Listen & compare': 'Слушайте и сравнивайте',
    'Export format': 'Формат файла',
    'Text (.txt)': 'Текст (.txt)',
    'Subtitles (.srt)': 'Субтитры (.srt)',
    'Subtitles (.vtt)': 'Субтитры (.vtt)',
    'Export text': 'Сохранить текст',
    'YOUR NEXT LISTEN STARTS HERE': 'ЗДЕСЬ ПОЯВИТСЯ РЕЗУЛЬТАТ',
    'A little less overlap.': 'Чуть меньше путаницы.',
    'A lot more understanding.': 'Гораздо больше понимания.',
    'Add a short recording to compare the original': 'Добавьте запись, чтобы сравнить оригинал',
    'with extracted sources and their transcripts.': 'с выделенными голосами и их текстом.',
    'Real audio': 'Настоящий звук',
    'Readable transcripts': 'Читаемый текст',
    'Side-by-side listening': 'Сравнение на слух',
    'WORKING THROUGH YOUR RECORDING': 'ОБРАБАТЫВАЕМ ВАШУ ЗАПИСЬ',
    'Preparing your audio': 'Готовим запись',
    'The first run may take longer while the models load.': 'Первый запуск может идти дольше: загружаются модели.',
    'Reconnecting to the speech engine…': 'Восстанавливаем связь с программой…',
    'Processing progress': 'Ход обработки',
    'Preparing': 'Подготовка',
    'Waiting for the engine': 'Ждём своей очереди',
    'Running speech models': 'Работают модели',
    'Keep this page open. Processing speed depends on your computer.': 'Не закрывайте страницу. Скорость зависит от вашего компьютера.',
    'Ready for a closer listen': 'Готово - можно слушать',
    'Machine-generated transcripts. Listen back to verify the words.': 'Текст создан автоматически. Сверяйте слова с записью.',
    'Review notes': 'На что обратить внимание',
    'Original recording': 'Исходная запись',
    'The audio you started with': 'Запись, с которой вы начали',
    'Waveform of original recording': 'Форма сигнала исходной записи',
    'ORIGINAL TRANSCRIPT': 'ТЕКСТ ИСХОДНОЙ ЗАПИСИ',
    'TRANSCRIPT': 'ТЕКСТ',
    'CONVERSATION': 'РАЗГОВОР',
    'Correct transcript': 'Исправить текст',
    'Finish correcting': 'Закончить исправление',
    'Save as check example': 'Сохранить как проверочный пример',
    'Click a speaker to give a name. Click a time to listen from there.': 'Нажмите на говорящего, чтобы дать имя. Нажмите на время, чтобы слушать с этого места.',
    'Fix the words in each turn and its language. Then keep the recording as a check example, or just export the corrected text.': 'Исправьте слова и язык в каждой реплике. Потом сохраните запись как проверочный пример или просто сохраните исправленный текст.',
    'Give this speaker a name': 'Дать имя этому говорящему',
    'Language of this turn': 'Язык этой реплики',
    'CHECK': 'ПРОВЕРЬТЕ',
    'AT THE SAME TIME': 'ОДНОВРЕМЕННО',
    'Two people spoke at the same time here; their voices were separated before recognition.': 'Здесь двое говорили одновременно; перед распознаванием голоса были разделены.',
    'A recognition or language score crossed a review threshold.': 'Оценка распознавания или языка пересекла порог: это место стоит проверить.',
    'No speech was transcribed in this recording.': 'В этой записи речь не распознана.',
    'No speech was transcribed for this track.': 'Для этой дорожки речь не распознана.',
    'EXTRACTED SOURCES': 'ВЫДЕЛЕННЫЕ ГОЛОСА',
    'Same moment. Separate listening.': 'Один момент. Каждый голос отдельно.',
    'Estimated speech source': 'Выделенный голос (оценка модели)',
    'Delete this result & audio': 'Удалить результат и запись',
    'SI-SDR improvement': 'Улучшение SI-SDR',
    "Mean separation improvement over this demo's noisy input, against its known clean sources. Audio quality, not transcription accuracy.": 'Среднее улучшение разделения относительно шумной записи примера, по известным чистым голосам. Это качество звука, а не точность текста.',
    'Demo comparison': 'Сравнение на примере',
    'Measured against the included demo reference recordings.': 'Измерено по эталонным записям готового примера.',
    'An unflagged segment may still contain errors.': 'Фрагмент без пометки тоже может содержать ошибки.',

    // ---- bottom strip, footer
    'Project scope': 'Рамки проекта',
    '01 / WHAT IT DOES': '01 / ЧТО УМЕЕТ',
    'Conversations in three languages & two-voice separation': 'Разговоры на трёх языках и разделение двух голосов',
    '02 / NEXT EXPLORATION': '02 / ЧТО ДАЛЬШЕ',
    'Real mixed-language speech & overlapping groups': 'Живая смешанная речь и группы, где говорят одновременно',
    '03 / RESEARCH DIRECTION': '03 / НАПРАВЛЕНИЕ ИССЛЕДОВАНИЙ',
    'Video-assisted speech recognition': 'Распознавание речи с помощью видео',
    'Read about project scope': 'Подробнее о рамках проекта',
    'CLEARTALK · FROM SIGNAL TO SENTENCE': 'CLEARTALK · ОТ СИГНАЛА К ПРЕДЛОЖЕНИЮ',
    'Inspectable models. Measured experiments. Reviewable transcripts.': 'Открытые модели. Измеренные эксперименты. Проверяемый текст.',

    // ---- project guide
    'THE PROJECT': 'ПРОЕКТ',
    'Close project guide': 'Закрыть описание проекта',
    'ClearTalk, explained.': 'ClearTalk простыми словами.',
    'A testable workspace for understanding speech when several people talk, in turn or at once.': 'Рабочая область, где можно проверить, как понимается речь, когда говорят несколько человек - по очереди или одновременно.',
    'What happens to a recording?': 'Что происходит с записью?',
    "SepFormer estimates two speech sources from the mixture. A Whisper speech recognizer then turns the mixture and each source into text: either the pretrained model or ClearTalk's fine-tuned one. You can listen to every result and compare the text.": 'SepFormer выделяет из смеси два голоса. Затем модель распознавания Whisper превращает в текст и смесь, и каждый голос: готовая модель или дообученная в ClearTalk. Каждый результат можно прослушать и сравнить текст.',
    'What is our own model?': 'Что такое наша собственная модель?',
    'Whisper small.en adapted with LoRA (parameter-efficient fine-tuning) to the kind of audio it receives here: voices extracted by the separator from noisy two-speaker mixtures, with their artefacts and 8 kHz bandwidth. Only small low-rank adapter matrices were trained, then merged, so it runs exactly as fast as the original.': 'Whisper small.en, дообученная методом LoRA (экономное дообучение малой части параметров) на том звуке, который она получает здесь: голоса, выделенные из шумных смесей двух говорящих, с их искажениями и полосой 8 кГц. Обучались только небольшие добавочные матрицы, потом их слили с моделью, поэтому она работает так же быстро, как исходная.',
    'How do we evaluate it?': 'Как мы её оцениваем?',
    'We measure word error rate (substitutions + deletions + insertions over reference words) on complete held-out utterances with official transcripts, from speakers never used in training. Checkpoints are chosen on validation data; the test set is scored once. Speech sources come from partitions of': 'Мы считаем долю ошибок в словах (замены + пропуски + вставки, делённые на число слов в эталоне) на целых отложенных высказываниях с официальным текстом, от дикторов, которых не было в обучении. Сохранённые состояния модели выбираются на проверочных данных; тестовый набор оценивается один раз. Речь взята из частей корпуса',
    ", a corpus of read English audiobook speech. These are controlled mixtures, not natural conversations; their results do not establish real-world accuracy. Cohort sizes and available measurements are shown in the active model's evidence panel.": ' - это английские аудиокниги, прочитанные вслух. Это контролируемые смеси, а не живые разговоры; их результаты не говорят о точности в реальной жизни. Размеры наборов и доступные замеры показаны в описании выбранной модели.',
    'What does Conversation mode do?': 'Что делает режим «Разговор»?',
    'It finds the stretches with speech, groups them by voice (speaker embeddings and clustering) so each turn gets a speaker label, detects the language of every turn among Kazakh, Russian and English, and sends the turn to a Whisper recognizer for that language. Ready recognizers are used as published; the project\'s own multilingual models, when installed, are Whisper models adapted with LoRA to the three languages and to recordings that change language. Which recognizer handles which language is decided by word error rate on validation data. If people switch language in the middle of a turn, choose "Languages change inside turns": every turn then goes to the recognizer trained for that. You can name the speakers, correct the text, export subtitles, and keep a corrected recording on this computer as a check example for measuring the recognizers on real speech.': 'Режим находит участки с речью и группирует их по голосам (числовые «отпечатки» голоса и их группировка), чтобы у каждой реплики была метка говорящего. Затем он определяет язык каждой реплики - казахский, русский или английский - и отдаёт реплику модели Whisper для этого языка. Готовые модели используются как есть; собственные многоязычные модели проекта, если они установлены, - это модели Whisper, дообученные методом LoRA на трёх языках и на записях со сменой языка. Какая модель отвечает за какой язык, решает доля ошибок в словах на проверочных данных. Если люди меняют язык посреди реплики, выберите «Языки меняются в реплике»: тогда каждую реплику записывает модель, обученная именно этому. Говорящим можно дать имена, текст можно исправить и сохранить как субтитры, а исправленную запись - оставить на этом компьютере как проверочный пример, чтобы измерять модели на живой речи.',
    'What are the limits?': 'Какие есть ограничения?',
    'The two-voice modes take short English audio and separation assumes two speakers; A/B are clip-local outputs, not identified people. Conversation mode separates people who talk at the same time only when "2 · at the same time" is chosen for a short clip (a demonstration, not measured on a test set), speaker labels are not identities, and a change of language in the middle of a turn can be missed. Mixed-language quality was measured on recordings assembled from single-language utterances, not on natural language switching. Video and lip reading are future research, not active features. Listen back: transcripts may miss or invent words.': 'Режимы с двумя голосами принимают короткие английские записи, а разделение рассчитано на двух говорящих; A и B - это метки внутри фрагмента, а не установленные личности. Режим «Разговор» разделяет одновременную речь, только если для короткой записи выбрать «2 · говорят одновременно» (демонстрация, на тестовом наборе не измерялась), метки говорящих - не личности, а смену языка посреди реплики можно пропустить. Качество на смешанной речи измерено на записях, собранных из высказываний на одном языке, а не на живом переключении языков. Видео и чтение по губам - это будущие исследования, сейчас их нет. Сверяйте с записью: в тексте могут пропадать или появляться лишние слова.',
    'Recording privacy & retention': 'Приватность и хранение записей',

    // ---- messages of the page
    'The server is not responding. Check that ClearTalk is running, then try again.': 'Программа не отвечает. Проверьте, что ClearTalk запущен, и попробуйте ещё раз.',
    'The recording could not be processed. Check the file and try again.': 'Не удалось обработать запись. Проверьте файл и попробуйте ещё раз.',
    'Processing did not complete. Please try a different recording.': 'Обработка не завершилась. Попробуйте другую запись.',
    'Processing finished without a result. Please try again.': 'Обработка закончилась без результата. Попробуйте ещё раз.',
    'The server could not start a processing job. Please try again.': 'Не удалось начать обработку. Попробуйте ещё раз.',
    'Choose an audio file such as WAV, MP3, M4A or FLAC. Video is not supported.': 'Выберите аудиофайл, например WAV, MP3, M4A или FLAC. Видео не поддерживается.',
    'This browser cannot record audio. Upload a file instead.': 'Этот браузер не умеет записывать звук. Загрузите файл.',
    'The microphone is not available. Allow it in the browser, or upload a file.': 'Микрофон недоступен. Разрешите его в браузере или загрузите файл.',
    'Opening your saved result': 'Открываем сохранённый результат',
    'ClearTalk is running, but its speech models still need to be installed. Follow the setup instructions, then refresh this page.': 'ClearTalk запущен, но модели ещё не установлены. Выполните шаги установки и обновите страницу.',
    'Permanently delete this result, its audio and transcript from this workspace? Download anything you want to keep first.': 'Удалить навсегда этот результат, его запись и текст? Сначала сохраните то, что хотите оставить.',
    'The processed result, audio and transcript were deleted from this workspace. This cannot be undone. Your original file on your device was not changed.': 'Результат, запись и текст удалены из рабочей области. Отменить это нельзя. Ваш исходный файл на устройстве не изменён.',
    'Keep this recording and its corrected transcript on this computer as a check example? Do this only if everyone in the recording agreed.': 'Сохранить эту запись и исправленный текст на этом компьютере как проверочный пример? Делайте это, только если все участники записи согласны.',
    'ClearTalk fine-tuned recognizer': 'Дообученная модель ClearTalk',
    'Measured on synthetic mixtures of read English audiobooks. Real meetings, accents and microphones can behave differently.': 'Измерено на искусственных смесях английских аудиокниг. Настоящие встречи, акценты и микрофоны могут вести себя иначе.',
    'CLEARTALK - TRANSCRIPT': 'CLEARTALK - РАСШИФРОВКА',
    'CLEARTALK TRANSCRIPT': 'CLEARTALK - РАСШИФРОВКА',
    'ORIGINAL RECORDING': 'ИСХОДНАЯ ЗАПИСЬ',
    'REVIEW NOTES': 'НА ЧТО ОБРАТИТЬ ВНИМАНИЕ',
    '(No speech transcribed)': '(речь не распознана)',
    'SOURCE': 'ИСТОЧНИК',
    'Machine-generated text. Verify against the audio.': 'Текст создан автоматически. Сверяйте с записью.',
    'LIMITATIONS': 'ОГРАНИЧЕНИЯ',
    '[check]': '[проверьте]',

    // ---- progress stages reported by the server
    'Validating the recording': 'Проверяем запись',
    'Waiting for the worker': 'Ждём своей очереди',
    'Preparing the recording': 'Готовим запись',
    'Preparing audio': 'Готовим запись',
    'Loading the speech models': 'Загружаем модели',
    'Finding speech and speaker turns': 'Ищем речь и реплики говорящих',
    'Separating people who talk at the same time': 'Разделяем людей, которые говорят одновременно',
    'Detecting the language of each turn': 'Определяем язык каждой реплики',
    'Preparing the transcript': 'Готовим текст',
    'Loading the English speech recognizer': 'Загружаем английскую модель распознавания',
    'Transcribing the original recording': 'Расшифровываем исходную запись',
    'Loading the two speaker separation model': 'Загружаем модель разделения двух голосов',
    'Separating two voices from the noisy mixture': 'Разделяем два голоса в шумной записи',
    'Separating two voices': 'Разделяем два голоса',
    'Only one voice found: transcribing as usual': 'Найден только один голос: расшифровываем как обычно',
    'Two people at the same time is a setting of Conversation mode.': 'Настройка «говорят одновременно» есть только в режиме «Разговор».',
    'Two people at the same time: the whole recording was split into two voices by the separator of the English two-voice mode, and each voice was transcribed on its own. This setting is a demonstration, not measured on a test set; words can be lost or given to the wrong voice.': 'Говорят одновременно: вся запись разделена на два голоса разделителем из английского режима, и каждый голос расшифрован отдельно. Это демонстрация, на тестовом наборе она не измерялась: слова могут пропасть или достаться не тому голосу.',
    'Two people at the same time was chosen, but the separator found only one voice: the recording was transcribed as usual.': 'Выбрано «говорят одновременно», но разделитель нашёл только один голос: запись расшифрована как обычно.',
    'Preparing transcripts and audio playback': 'Готовим текст и звук',
    'Complete': 'Готово',
    'Processing interrupted': 'Обработка прервана',
    'Processing stopped': 'Обработка остановлена',

    // ---- errors reported by the server
    'The access token is incorrect.': 'Неверный ключ доступа.',
    'The demo recording is not installed yet. Run the setup script.': 'Пример записи ещё не установлен. Запустите установку.',
    'The demo recording is not installed yet.': 'Пример записи ещё не установлен.',
    'The multilingual demo recording is not installed.': 'Многоязычный пример записи не установлен.',
    'Audio file not found.': 'Аудиофайл не найден.',
    'Mode must be separate, transcribe or conversation.': 'Неизвестный режим обработки.',
    'Choose either the demo or an uploaded file.': 'Выберите что-то одно: пример или свой файл.',
    'Choose an audio file or use the demo.': 'Выберите аудиофайл или загрузите пример.',
    'Model must be auto or one of the multilingual recognizers.': 'Неизвестная модель распознавания.',
    'Model must be baseline or cleartalk.': 'Неизвестная модель распознавания.',
    'Language must be auto, mixed, kk, ru or en.': 'Неизвестный язык.',
    'The number of speakers must be between 1 and 8, or 0 to find it automatically.': 'Число говорящих - от 1 до 8, или 0, чтобы найти его автоматически.',
    'The multilingual speech models are not installed yet. Complete setup and restart ClearTalk.': 'Многоязычные модели ещё не установлены. Завершите установку и перезапустите ClearTalk.',
    'The separation model is not installed yet. Complete setup and restart ClearTalk.': 'Модель разделения голосов ещё не установлена. Завершите установку и перезапустите ClearTalk.',
    'The ClearTalk fine-tuned recognizer is not installed. Select the pretrained model.': 'Дообученная модель ClearTalk не установлена. Выберите готовую модель.',
    'The speech recognizer is not installed yet. Complete setup and restart ClearTalk.': 'Модель распознавания ещё не установлена. Завершите установку и перезапустите ClearTalk.',
    'Supported files: WAV, MP3, M4A, FLAC, OGG, AAC, MP4 and WebM.': 'Подходят файлы WAV, MP3, M4A, FLAC, OGG, AAC, MP4 и WebM.',
    'This recording was not found or has expired.': 'Запись не найдена, или срок её хранения истёк.',
    'This recording was not found or has expired. Open it in the browser that created it.': 'Запись не найдена, или срок её хранения истёк. Откройте её в том браузере, где она была создана.',
    'The transcript is not ready yet.': 'Текст ещё не готов.',
    'Check examples can only be kept on a local installation.': 'Проверочные примеры можно сохранять только при работе на своём компьютере.',
    'Confirm that everyone in the recording agreed to it being kept.': 'Подтвердите, что все участники записи согласны на её сохранение.',
    'Only a finished conversation can be kept as a check example.': 'Сохранить как проверочный пример можно только готовую расшифровку разговора.',
    'The corrected transcript has no text or a turn that ends before it starts.': 'В исправленном тексте нет слов, или реплика заканчивается раньше, чем начинается.',
    'This file does not contain an audio track.': 'В этом файле нет звуковой дорожки.',
    'This audio could not be decoded. Try a WAV, MP3, M4A or FLAC file.': 'Не удалось прочитать звук. Попробуйте файл WAV, MP3, M4A или FLAC.',
    'Please provide at least a quarter-second of audio.': 'Нужна запись длиной хотя бы четверть секунды.',
    'This file contains invalid audio samples.': 'В файле повреждённые звуковые данные.',
    'This recording is silent. Please choose a recording with speech.': 'В записи тишина. Выберите запись с речью.',
    'The processing queue is full. Please wait for a recording to finish.': 'Очередь обработки заполнена. Дождитесь, пока закончится текущая запись.',
    'Recording storage is full. Delete an old result or wait for its retention period to expire.': 'Хранилище записей заполнено. Удалите старый результат или дождитесь, пока истечёт срок его хранения.',
    'Please wait for processing to finish before deleting this recording.': 'Дождитесь конца обработки, прежде чем удалять запись.',
    'The request exceeds the configured upload limit.': 'Файл больше допустимого размера.',
    'The request could not be processed.': 'Не удалось обработать запрос.',
    'The service restarted before this recording finished. Please upload it again.': 'Программа перезапустилась до конца обработки. Загрузите запись ещё раз.',
    'There was not enough memory. Try a shorter recording.': 'Не хватило памяти. Попробуйте запись короче.',
    'A speech model this recording needs is not installed correctly. Complete setup and restart ClearTalk.': 'Модель, нужная для этой записи, установлена неправильно. Завершите установку и перезапустите ClearTalk.',
    'Audio processing failed. Try a shorter WAV recording. Technical details are available in the service logs.': 'Обработка не удалась. Попробуйте запись короче в формате WAV. Технические подробности есть в журнале программы.',

    // ---- notes attached to a result by the server
    'Machine-generated transcript in Kazakh, Russian or English; it can contain errors. Check the audio before relying on it.': 'Текст на казахском, русском или английском создан автоматически и может содержать ошибки. Сверяйте с записью, прежде чем полагаться на него.',
    'Turn times are approximate. Review flags use simple score thresholds, not calibrated confidence.': 'Время реплик приблизительное. Пометки «проверьте» ставятся по простым порогам оценок, это не точная мера уверенности.',
    'Where two people spoke at the same time, their voices were separated and written as two turns marked for checking; words can still be lost or given to the wrong speaker.': 'Там, где двое говорили одновременно, голоса разделены и записаны двумя репликами с пометкой «проверить»; слова всё равно могут пропасть или достаться не тому говорящему.',
    'Short quiet stretches that sounded like background voices or noise. They are not in the transcript or the exports.': 'Короткие тихие фрагменты, похожие на фоновые голоса или шум. В расшифровку и в файлы для скачивания они не входят.',
    'People speaking at the same time are not separated in this mode: overlapping words can be lost or attributed to one speaker.': 'Одновременную речь этот режим не разделяет: слова, сказанные одновременно, могут пропасть или достаться одному говорящему.',
    'Speaker 1, Speaker 2, … are labels inside this recording, not identified people. Short interjections can be attached to the wrong speaker.': '«Говорящий 1», «Говорящий 2» и так далее - это метки внутри записи, а не установленные личности. Короткие реплики могут достаться не тому говорящему.',
    'Speaker labels were not computed for this recording: all text is shown as one speaker.': 'Метки говорящих для этой записи не определялись: весь текст показан как один говорящий.',
    'The language was detected turn by turn among Kazakh, Russian and English. A language change in the middle of a turn, or a very short turn, can be written in the wrong language.': 'Язык определялся для каждой реплики: казахский, русский или английский. Если язык меняется посреди реплики или реплика очень короткая, она может быть записана не на том языке.',
    'Mixed-language setting: every turn was written by the recognizer trained on recordings that change language. It was trained and measured on assembled recordings, not on one person switching language inside a sentence; check such places against the audio.': 'Настройка «Языки меняются в реплике»: каждую реплику записала модель, обученная на записях со сменой языка. Её учили и проверяли на собранных записях, а не на речи одного человека, который меняет язык внутри фразы; такие места сверяйте с записью.',
    'The ClearTalk multilingual recognizer was fine-tuned on read sentences and short phrases; its measured gains do not guarantee better text on every recording.': 'Многоязычную модель ClearTalk дообучали на прочитанных предложениях и коротких фразах; измеренный выигрыш не гарантирует лучший текст на каждой записи.',
    'No speech was transcribed. Quiet or heavily distorted speech can be missed.': 'Речь не распознана. Тихая или сильно искажённая речь может быть пропущена.',
    'English speech only. Transcripts can contain errors; check the audio before relying on them.': 'Только английская речь. В тексте могут быть ошибки; сверяйте с записью, прежде чем полагаться на него.',
    'Segment times are approximate. Review flags use simple score thresholds, not calibrated confidence.': 'Время фрагментов приблизительное. Пометки «проверьте» ставятся по простым порогам оценок, это не точная мера уверенности.',
    'Speech detection reduces text on non-speech audio, but can miss very quiet or distorted words.': 'Поиск речи убирает лишний текст там, где речи нет, но может пропустить очень тихие или искажённые слова.',
    'The ClearTalk recognizer was fine-tuned on synthetic mixtures of read English audiobooks. Its measured gains do not guarantee better text on every recording; compare with the pretrained model.': 'Модель ClearTalk дообучали на искусственных смесях английских аудиокниг. Измеренный выигрыш не гарантирует лучший текст на каждой записи; сравните с готовой моделью.',
    'The separator assumes exactly two simultaneous speakers. A single speaker or more speakers can produce misleading tracks.': 'Разделение рассчитано ровно на двух одновременно говорящих. С одним или с большим числом говорящих дорожки могут вводить в заблуждение.',
    'Source A and Source B are arbitrary labels for this clip, not identified people.': '«Источник A» и «Источник B» - условные метки для этого фрагмента, а не установленные личности.',
    'Playback levels are normalized separately. Separation may leave interference or remove words.': 'Громкость дорожек выровнена по отдельности. После разделения могут остаться помехи или пропасть слова.',
    'Transcription-only mode does not separate voices or assign speaker labels.': 'Режим «Только расшифровать» не разделяет голоса и не ставит метки говорящих.',
    'The demo audio quality metric could not be calculated.': 'Не удалось посчитать оценку качества звука для примера.',
    'The recording is empty or digitally silent; no words were transcribed.': 'Запись пустая, или в ней полная тишина; слова не распознаны.',
    'No speech was detected. Quiet or heavily distorted speech can be missed; listen to the recording before concluding that it contains no speech.': 'Речь не найдена. Тихая или сильно искажённая речь может быть пропущена; прослушайте запись, прежде чем решить, что речи в ней нет.',
    'Speech activity was detected, but no words were transcribed. Listen to the recording to review it.': 'Речь найдена, но слова не распознаны. Прослушайте запись.',

    // ---- what the model cards say
    'ClearTalk multilingual: Whisper small adapted with LoRA to Kazakh, Russian, English and to recordings that change language.': 'ClearTalk многоязычная: Whisper small, дообученная методом LoRA на казахском, русском, английском и на записях со сменой языка.',
    'ClearTalk large multilingual: the Kazakh Whisper large-v3-turbo adapted with LoRA (decoder only) to Kazakh, Russian, English and to recordings that change language.': 'ClearTalk большая многоязычная: казахская Whisper large-v3-turbo, дообученная методом LoRA (только часть, которая пишет текст) на казахском, русском, английском и на записях со сменой языка.',
    'Mixed recordings are assembled from single-language utterances: every language change is also a change of speaker. Real conversations, accents and microphones can behave differently.': 'Записи со сменой языка собраны из высказываний на одном языке: каждая смена языка - это ещё и смена говорящего. Живые разговоры, акценты и микрофоны могут вести себя иначе.',
  };

  const REVIEW_NOTE = 'Check (\\d+\\.\\d)-(\\d+\\.\\d)s: an ASR score crossed a review threshold\\. This is a heuristic, not a calibrated confidence estimate\\.';
  const LANGUAGE_NAME = '(?:Kazakh|Russian|English|Mixed)';
  const CLOCK = '\\d+:\\d\\d(?::\\d\\d)?';
  const TIME = `(?:${CLOCK}|\\d+\\.\\ds)`;

  /* Sentences with numbers, names or times in them. Each pattern must match the whole text. */
  const PATTERNS = [
    [/^Speaker (\d+)$/, m => `Говорящий ${m[1]}`],
    [/^Source ([A-Z])$/, m => `Источник ${m[1]}`],
    [/^(\d+\.\d)s$/, m => duration(m[0])],
    [/^([+-]?\d+\.\d\d) dB$/, m => `${decimal(m[1])} дБ`],
    [new RegExp(`^${LANGUAGE_NAME.toUpperCase()}(?: \\+ ${LANGUAGE_NAME.toUpperCase()})*$`), m => m[0].split(' + ').map(name => language(capital(name.toLowerCase())).toUpperCase()).join(' + ')],
    [/^WAV, MP3, M4A & more · up to (\d+) (min|sec) \/ (\d+) MB$/, m => `WAV, MP3, M4A и другие · до ${m[1]} ${m[2] === 'min' ? 'мин' : 'с'} / ${m[3]} МБ`],
    [/^This file is too large for this mode\. Choose an audio file smaller than (\d+) MB\.$/, m => `Файл слишком большой для этого режима. Выберите аудиофайл меньше ${m[1]} МБ.`],
    [new RegExp(`^This recording is (${CLOCK}) long\\. Conversation mode takes up to (\\d+) minutes\\.$`), m => `Длина записи - ${m[1]}. Режим «Разговор» принимает записи до ${m[2]} мин.`],
    [/^This recording is (\d+\.\ds) long\. This mode takes clips of (\d+) seconds or less; use Conversation mode for long recordings\.$/, m => `Длина записи - ${duration(m[1])}. Этот режим принимает фрагменты не длиннее ${m[2]} с; для длинных записей выберите режим «Разговор».`],
    [/^Please trim the recording to ([\d.]+) (seconds|minutes) or less\.$/, m => `Сократите запись до ${decimal(m[1])} ${m[2] === 'minutes' ? 'мин' : 'с'} или меньше.`],
    [/^Please choose a file smaller than (\d+) MB\.$/, m => `Выберите файл меньше ${m[1]} МБ.`],
    [/^(.+) is not installed\. Select another recognizer\.$/, m => `Модель «${models(m[1])}» не установлена. Выберите другую.`],
    [/^The server returned an unexpected response \((\d+)\)\. Please try again\.$/, m => `Программа ответила неожиданно (код ${m[1]}). Попробуйте ещё раз.`],
    [/^Request failed \((\d+)\)\.$/, m => `Запрос не выполнен (код ${m[1]}).`],
    [/^(.+?)( \(ours\))?( · not installed)?$/, m => Object.prototype.hasOwnProperty.call(MODELS, m[1]) || /^(?:Whisper|Kazakh Whisper|ClearTalk) /.test(m[1]) && (m[2] || m[3])
      ? `${models(m[1])}${m[2] ? ' (наша)' : ''}${m[3] ? ' · не установлена' : ''}` : null],
    [/^Every turn → (.+?)\. Use this when people switch language in the middle of a turn\.( Speaker labels are not installed: all text is shown as one speaker\.)?$/,
      m => `Каждая реплика → ${models(m[1])}. Выбирайте это, когда люди меняют язык посреди реплики.${m[2] ? NO_LABELS : ''}`],
    [new RegExp(`^(${LANGUAGE_NAME} → .+?)\\.( The language is detected turn by turn\\.)?( Speaker labels are not installed: all text is shown as one speaker\\.)?$`),
      m => `${m[1].split('; ').map(route => { const [name, model] = route.split(' → '); return `${language(name)} → ${models(model)}`; }).join('; ')}.${m[2] ? ' Язык определяется для каждой реплики.' : ''}${m[3] ? NO_LABELS : ''}`],
    [/^(\d+\.\d\d) MB · selected audio$/, m => `${decimal(m[1])} МБ · выбранная запись`],
    [new RegExp(`^(${TIME}) · (\\d+\\.\\d\\d) MB$`), m => `${duration(m[1])} · ${decimal(m[2])} МБ`],
    [new RegExp(`^(${TIME}) · recorded here$`), m => `${duration(m[1])} · записано здесь`],
    [new RegExp(`^Stop recording · (${CLOCK})$`), m => `Остановить запись · ${m[1]}`],
    [/^Short quiet stretches left out of the transcript as background voices or noise: (\d+)\. They are listed below it\.$/,
      m => `Короткие тихие фрагменты, не вошедшие в расшифровку как фоновые голоса или шум: ${m[1]}. Они показаны под ней.`],
    [/^Left out as background: (\d+)$/, m => `Не вошло в расшифровку как фон: ${m[1]}`],
    [/^Transcribing turn (\d+) of (\d+)$/, m => `Расшифровываем реплику ${m[1]} из ${m[2]}`],
    [/^Transcribing voice (\d+) of 2$/, m => `Расшифровываем голос ${m[1]} из 2`],
    [/^Transcribing Source ([A-Z])$/, m => `Расшифровываем источник ${m[1]}`],
    [new RegExp(`^(.*) · (${CLOCK}) · (${LANGUAGE_NAME}(?:, ${LANGUAGE_NAME})*)$`), m => `${speaker(m[1])} · ${m[2]} · ${languages(m[3], ', ')}`],
    [new RegExp(`^(\\d+) speakers? · (\\d+) turns?(?: · (.+))?$`), m => `${m[1]} ${plural(m[1], 'говорящий', 'говорящих', 'говорящих')} · ${m[2]} ${plural(m[2], 'реплика', 'реплики', 'реплик')}${m[3]
      ? ` · ${m[3].split(', ').map(item => item.replace(new RegExp(`^(${LANGUAGE_NAME}(?: \\+ ${LANGUAGE_NAME})*) (${CLOCK})$`), (all, names, time) => `${languages(names, ' + ')} ${time}`)).join(', ')}` : ''}`],
    [new RegExp(`^Play from (${CLOCK})$`), m => `Слушать с ${m[1]}`],
    [/^Name for (.+) \(leave empty to go back to the label\)$/, m => `Имя для «${speaker(m[1])}» (оставьте пустым, чтобы вернуть метку)`],
    [/^Saved\. Check examples on this computer: (\d+)\.$/, m => `Сохранено. Проверочных примеров на этом компьютере: ${m[1]}.`],
    [new RegExp(`^(${TIME}) audio(?: · (${TIME}) processing)?$`), m => `Запись ${duration(m[1])}${m[2] ? ` · обработка ${duration(m[2])}` : ''}`],
    [/^Review notes \((\d+)\)$/, m => `На что обратить внимание (${m[1]})`],
    [/^(?:Processed with (.+)\. )?An unflagged segment may still contain errors\.$/, m => `${m[1]
      ? `Обработано моделями: ${models(m[1]).replace(/ · speech detection/g, ' · с поиском речи').replace(/speaker labels from /g, 'метки говорящих - ')}. ` : ''}Фрагмент без пометки тоже может содержать ошибки.`],
    [/^(Waveform of|Listen to) (.+)$/, m => `${m[1] === 'Listen to' ? 'Слушать:' : 'Форма сигнала:'} ${source(m[2])}`],
    [new RegExp(`^(?:${REVIEW_NOTE}\\s?)+$`), m => m[0].replace(new RegExp(REVIEW_NOTE, 'g'), (all, start, end) => `Проверьте ${decimal(start)}-${decimal(end)} с: оценка распознавания пересекла порог. Это простое правило, а не точная мера уверенности.`)],
    [/^Uploaded audio is processed on the machine hosting this workspace\. Results are available to this browser session and automatically expire after ([\d.]+) hours; you can delete a finished result sooner\. The server owner can access stored files\. Only upload recordings you have permission to process\.$/,
      m => `Загруженные записи обрабатываются на компьютере, где запущена эта рабочая область. Результаты видны только в этом браузере и сами удаляются через ${decimal(m[1])} ч; готовый результат можно удалить раньше. Владелец компьютера имеет доступ к сохранённым файлам. Загружайте только те записи, которые вам разрешено обрабатывать.`],
    [/^Results are private to this browser session and expire after (\d+) hours\. The server owner can access stored recordings\. Only upload audio you have permission to process\.$/,
      m => `Результаты видны только в этом браузере и удаляются через ${m[1]} ч. Владелец компьютера имеет доступ к сохранённым записям. Загружайте только те записи, которые вам разрешено обрабатывать.`],

    // ---- model cards
    [/^([\d,]+) trained LoRA parameters \(([\d.]+)% of ([\d,]+)\); the merged model has the same size and speed as the pretrained one\.$/,
      m => `${thousands(m[1])} обученных параметров LoRA (${decimal(m[2])}% от ${thousands(m[3])}); после слияния модель того же размера и той же скорости, что и готовая.`],
    [/^([\d,]+) trained LoRA parameters \(([\d.]+)% of ([\d,]+)\); merged, so it runs as fast as (.+)\.$/,
      m => `${thousands(m[1])} обученных параметров LoRA (${decimal(m[2])}% от ${thousands(m[3])}); они слиты с моделью, поэтому она работает так же быстро, как ${system(m[4])}.`],
    [/^Trained on ([\d,]+) separated LibriSpeech tracks from (\d+) speakers; validation and test speakers were never seen in training\.$/,
      m => `Обучена на ${thousands(m[1])} разделённых дорожках LibriSpeech от ${m[2]} дикторов; дикторов из проверочного и тестового наборов в обучении не было.`],
    [/^Trained on ([\d,]+) examples \(([\d,]+) of them mixed-language recordings assembled from FLEURS, Golos and KSC2 utterances\), one pass, on a GTX 1660 Ti\.$/,
      m => `Обучена на ${thousands(m[1])} примерах (из них ${thousands(m[2])} - записи со сменой языка, собранные из высказываний FLEURS, Golos и KSC2), один проход, на GTX 1660 Ti.`],
    [/^(Validation|Final test), (.+?) \(([\d,]+)\): (?:word error rate )?([\d.]+)% (.+?) → ([\d.]+)% (.+?)\.( Lower is better\.)?$/,
      m => `${m[1] === 'Validation' ? 'Проверка' : 'Итоговый тест'}, ${SETS[m[2]] || m[2]} (${thousands(m[3])}): доля ошибок в словах ${decimal(m[4])}% ${system(m[5])} → ${decimal(m[6])}% ${system(m[7])}.${m[8] ? ' Чем меньше, тем лучше.' : ''}`],
    [/^(Validation|Final test), (.+?): word error rate ([\d.]+)% pretrained → ([\d.]+)% ClearTalk on the same ([\d,]+) (mixtures|utterances)\. Lower is better\.$/,
      m => `${m[1] === 'Validation' ? 'Проверка' : 'Итоговый тест'}, ${SETS[m[2]] || m[2]}: доля ошибок в словах ${decimal(m[3])}% готовая → ${decimal(m[4])}% ClearTalk на одних и тех же ${thousands(m[5])} ${m[6] === 'mixtures' ? 'смесях' : 'высказываниях'}. Чем меньше, тем лучше.`],
  ];

  function lookup(core) {
    if (Object.prototype.hasOwnProperty.call(EXACT, core)) return EXACT[core];
    for (const [pattern, build] of PATTERNS) {
      const match = pattern.exec(core);
      if (match) { const built = build(match); if (built != null) return built; }
    }
    return null;
  }

  /* Russian for one piece of interface text, or null when there is nothing to translate. */
  function translate(text) {
    if (typeof text !== 'string' || !/[A-Za-z]/.test(text)) return null;
    const parts = /^(\s*)([\s\S]*?)(\s*)$/.exec(text);
    const core = parts[2].replace(/\s+/g, ' ');
    const found = lookup(core);
    return found == null ? null : parts[1] + found + parts[3];
  }

  if (typeof document === 'undefined') {  // loaded by the tests
    if (typeof module !== 'undefined') module.exports = { translate, EXACT };
    return;
  }

  const ATTRIBUTES = ['placeholder', 'aria-label', 'title'];
  const UNTOUCHED = '[translate="no"], .turn-text, .transcript:not(.unavailable), [contenteditable="true"], script, style';
  const STORAGE_KEY = 'cleartalk-language';
  const shownText = new WeakMap();       // text node -> { source, shown }
  const shownAttributes = new WeakMap(); // element -> { attribute: { source, shown } }
  const pageTitle = document.title;
  let current = 'en';

  const text = (value) => current === 'ru' ? (translate(String(value)) ?? String(value)) : String(value);

  function showText(node) {
    const parent = node.parentElement;
    if (!parent || parent.closest(UNTOUCHED)) return;
    const known = shownText.get(node);
    const value = node.nodeValue;
    const original = known && value === known.shown ? known.source : value;
    const shown = text(original);
    shownText.set(node, { source: original, shown });
    if (value !== shown) node.nodeValue = shown;
  }

  function showAttribute(element, name) {
    if (!element.hasAttribute(name)) return;
    const all = shownAttributes.get(element) || {};
    const known = all[name];
    const value = element.getAttribute(name);
    const original = known && value === known.shown ? known.source : value;
    const shown = text(original);
    all[name] = { source: original, shown };
    shownAttributes.set(element, all);
    if (value !== shown) element.setAttribute(name, shown);
  }

  function showTree(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) showText(node);
    const selector = ATTRIBUTES.map(name => `[${name}]`).join(',');
    if (root.matches?.(selector)) ATTRIBUTES.forEach(name => showAttribute(root, name));
    root.querySelectorAll(selector).forEach(element => ATTRIBUTES.forEach(name => showAttribute(element, name)));
  }

  function apply(next) {
    current = next === 'ru' ? 'ru' : 'en';
    document.documentElement.lang = current;
    document.title = text(pageTitle);
    showTree(document.body);
    const button = document.getElementById('lang-switch');
    if (button) button.textContent = current === 'ru' ? 'English' : 'Русский';
  }

  new MutationObserver(changes => {
    if (current === 'en') return;
    changes.forEach(change => {
      if (change.type === 'characterData') showText(change.target);
      else if (change.type === 'attributes') showAttribute(change.target, change.attributeName);
      else change.addedNodes.forEach(node => { if (node.nodeType === Node.TEXT_NODE) showText(node); else if (node.nodeType === Node.ELEMENT_NODE) showTree(node); });
    });
  }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRIBUTES });

  // Questions the browser asks in its own windows.
  ['confirm', 'prompt', 'alert'].forEach(name => {
    const ask = window[name].bind(window);
    window[name] = (message, ...rest) => ask(text(message ?? ''), ...rest);
  });

  const button = document.getElementById('lang-switch');
  if (button) {
    button.setAttribute('translate', 'no');
    button.addEventListener('click', () => {
      apply(current === 'ru' ? 'en' : 'ru');
      try { localStorage.setItem(STORAGE_KEY, current); } catch { /* the choice lasts for this page only */ }
    });
  }

  let saved = null;
  try { saved = localStorage.getItem(STORAGE_KEY); } catch { /* no storage: follow the browser */ }
  const browser = (navigator.languages?.[0] || navigator.language || 'en').toLowerCase();
  apply(saved === 'ru' || saved === 'en' ? saved : /^(ru|kk|be|uk)\b/.test(browser) ? 'ru' : 'en');

  window.ClearTalkLanguage = { text, set: apply, get current() { return current; } };
})();
