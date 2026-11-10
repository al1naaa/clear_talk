/* ClearTalk's interface renders only results returned by the inference server. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = { file: null, fileSeconds: null, demo: null, conversationDemo: null, useDemo: false, mode: 'separate', model: 'baseline', language: 'auto', speakers: 0, twoVoices: false, conversationModel: 'auto', speakerNames: {}, edits: {}, editing: false, locked: false, busy: false, health: null, result: null, jobId: null, jobStatus: null, generation: 0, objectUrl: null };
  const elements = {
    file: $('audio-file'), drop: $('drop-zone'), demo: $('demo-button'), run: $('run-button'), export: $('export-button'),
    selected: $('selected-file'), error: $('input-error'), empty: $('empty-state'), progress: $('progress-state'), results: $('results-content'),
  };
  const LANGUAGE_CODES = ['kk', 'ru', 'en'];
  const LANGUAGE_LABELS = { kk: 'Kazakh', ru: 'Russian', en: 'English', mixed: 'Mixed' };
  const SPEAKER_COLORS = ['#c5946c', '#6f9c8f', '#7b8fc0', '#b98bb0', '#a3a05a', '#d0806f', '#5f9bb3', '#8f8f8f'];
  const show = (element, visible) => element.classList.toggle('hidden', !visible);
  const seconds = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}s` : '';
  const clock = (value) => {
    const total = Math.max(0, Math.floor(Number(value) || 0));
    const pad = (number) => String(number).padStart(2, '0');
    return total >= 3600 ? `${Math.floor(total / 3600)}:${pad(Math.floor(total % 3600 / 60))}:${pad(total % 60)}` : `${Math.floor(total / 60)}:${pad(total % 60)}`;
  };
  const displayName = (model) => typeof model === 'string' ? model : model?.name || '';
  // Files written by the page (exports) use the interface language chosen in i18n.js.
  const inLanguage = (text) => window.ClearTalkLanguage ? window.ClearTalkLanguage.text(text) : text;
  const recovery = document.createElement('div');
  recovery.id = 'job-recovery'; recovery.className = 'hidden';
  const recoveryText = document.createElement('p');
  const checkRecording = document.createElement('button');
  checkRecording.type = 'button'; checkRecording.className = 'export-button'; checkRecording.textContent = 'Check this recording';
  const deleteRecordingButton = document.createElement('button');
  deleteRecordingButton.type = 'button'; deleteRecordingButton.className = 'delete-result'; deleteRecordingButton.textContent = 'Delete retained recording';
  recovery.append(recoveryText, checkRecording, deleteRecordingButton);
  elements.error.after(recovery);

  function rememberJob(id, status) {
    state.jobId = id; state.jobStatus = status;
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('job', id);
    else url.searchParams.delete('job');
    window.history.replaceState(null, '', url);
  }

  function syncRecovery() {
    const visible = Boolean(state.jobId && !state.result && !state.busy);
    const finished = ['done', 'error'].includes(state.jobStatus);
    show(recovery, visible);
    checkRecording.disabled = state.busy;
    deleteRecordingButton.disabled = state.busy || !finished;
    recoveryText.textContent = finished
      ? 'This recording is still retained privately for your browser. You can delete its audio now.'
      : 'This recording may still be processing. Its link is saved in this page address. Check again to recover the result; active recordings cannot be deleted.';
  }

  function reportError(message) {
    elements.error.textContent = message || '';
    show(elements.error, Boolean(message));
  }

  function notice(message) {
    $('global-notice').textContent = message || '';
    show($('global-notice'), Boolean(message));
  }

  async function request(url, options) {
    let response;
    try { response = await fetch(url, options); }
    catch { throw new Error('The server is not responding. Check that ClearTalk is running, then try again.'); }
    if (response.status === 204) return null;
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The server returned an unexpected response (${response.status}). Please try again.`); }
    if (!response.ok) {
      const detail = data.detail || data.error || `Request failed (${response.status}).`;
      throw new Error(typeof detail === 'string' ? detail : 'The recording could not be processed. Check the file and try again.');
    }
    return data;
  }

  /* Recognizers a conversation job needs; mirrors the server's check. */
  function conversationNeeds() {
    const multilingual = state.health?.multilingual;
    if (!multilingual) return [];
    if (state.language === 'mixed') {
      const chosen = multilingual.recognizers?.[state.conversationModel];
      const coversAll = state.conversationModel !== 'auto' && LANGUAGE_CODES.every(code => chosen?.languages?.includes(code));
      return [...new Set([multilingual.detector, coversAll ? state.conversationModel : multilingual.mixed_recognizer])];
    }
    const needed = new Set(state.language === 'auto' ? [multilingual.detector] : []);
    (state.language === 'auto' ? LANGUAGE_CODES : [state.language]).forEach(language => {
      const chosen = multilingual.recognizers?.[state.conversationModel];
      needed.add(state.conversationModel !== 'auto' && chosen?.languages?.includes(language) ? state.conversationModel : multilingual.routes?.[language]);
    });
    return [...needed];
  }

  function readyForMode() {
    if (state.locked) return false;
    if (!state.health) return true;
    if (state.mode === 'conversation') {
      const recognizers = state.health.multilingual?.recognizers;
      return Boolean(recognizers) && (!state.twoVoices || state.health.models?.separator?.available)
        && conversationNeeds().every(key => recognizers[key]?.available);
    }
    const { models } = state.health;
    if (models?.[state.model]?.available === false) return false;
    if (state.mode === 'separate' && models?.separator?.available === false) return false;
    return true;
  }

  function modeLimits() {
    const limits = state.health?.limits || {};
    return state.mode === 'conversation'
      ? { seconds: state.twoVoices ? Math.min(30, (limits.max_long_minutes || 60) * 60) : (limits.max_long_minutes || 60) * 60, megabytes: limits.max_long_upload_mb || 300 }
      : { seconds: limits.max_duration_seconds || 20, megabytes: limits.max_upload_mb || 20 };
  }

  function limitText(limit) {
    return `${limit.seconds > 120 ? `${Math.round(limit.seconds / 60)} min` : `${limit.seconds} sec`} / ${limit.megabytes} MB`;
  }

  /* A selected file must fit the current mode; switching mode re-checks it. */
  function selectionProblem() {
    if (!state.file) return '';
    const limit = modeLimits();
    if (state.file.size > limit.megabytes * 1024 * 1024) return `This file is too large for this mode. Choose an audio file smaller than ${limit.megabytes} MB.`;
    if (Number.isFinite(state.fileSeconds) && state.fileSeconds > limit.seconds + .1) {
      return state.mode === 'conversation' && !state.twoVoices
        ? `This recording is ${clock(state.fileSeconds)} long. Conversation mode takes up to ${Math.round(limit.seconds / 60)} minutes.`
        : `This recording is ${seconds(state.fileSeconds)} long. This mode takes clips of ${limit.seconds} seconds or less; use Conversation mode for long recordings.`;
    }
    return '';
  }

  function currentDemo() { return state.mode === 'conversation' ? (state.twoVoices ? state.overlapDemo : state.conversationDemo) : state.demo; }

  function syncConversationOptions() {
    const multilingual = state.health?.multilingual;
    const conversation = state.mode === 'conversation';
    show($('conversation-options'), conversation);
    show($('model-picker'), !conversation);
    ['language-select', 'speakers-select', 'conversation-model-select'].forEach(id => { $(id).disabled = state.busy || state.locked; });
    $('language-select').value = state.language;
    $('speakers-select').value = state.twoVoices ? 'two-voices' : String(state.speakers);
    if (multilingual?.recognizers) {
      const select = $('conversation-model-select');
      const options = [['auto', 'Best measured model for each language', true],
        ...Object.entries(multilingual.recognizers).map(([key, item]) => [key, `${item.name}${key.startsWith('cleartalk') ? ' (ours)' : ''}${item.available ? '' : ' · not installed'}`, item.available])];
      if (select.options.length !== options.length) {
        select.replaceChildren(...options.map(([value, label, available]) => { const option = document.createElement('option'); option.value = value; option.textContent = label; option.disabled = !available; return option; }));
      }
      if (!options.some(([value, , available]) => value === state.conversationModel && available)) state.conversationModel = 'auto';
      select.value = state.conversationModel;
      const name = key => multilingual.recognizers[key]?.name || key;
      const shown = state.language === 'auto' ? LANGUAGE_CODES : [state.language];
      const chosen = multilingual.recognizers[state.conversationModel];
      const routes = shown.map(language => `${LANGUAGE_LABELS[language]} → ${name(state.conversationModel !== 'auto' && chosen?.languages?.includes(language) ? state.conversationModel : multilingual.routes[language])}`);
      const mixedKey = conversationNeeds().find(key => key !== multilingual.detector) || multilingual.mixed_recognizer;
      $('conversation-model-description').textContent = state.language === 'mixed'
        ? `Every turn → ${name(mixedKey)}. Use this when people switch language in the middle of a turn.${multilingual.speaker_labels ? '' : ' Speaker labels are not installed: all text is shown as one speaker.'}`
        : `${routes.join('; ')}.${state.language === 'auto' ? ' The language is detected turn by turn.' : ''}${multilingual.speaker_labels ? '' : ' Speaker labels are not installed: all text is shown as one speaker.'}`;
      const cards = ['cleartalk-large', 'cleartalk-kk', 'cleartalk'].map(key => multilingual.recognizers[key]).filter(item => item?.available)
        .map(item => Array.isArray(item.model_card?.evidence) ? item.model_card.evidence.filter(line => typeof line === 'string' && line.trim()) : [])
        .filter(lines => lines.length);
      show($('multilingual-details'), cards.length > 0);
      $('multilingual-details').querySelector('summary').textContent = cards.length > 1 ? 'About our multilingual models' : 'About our multilingual model';
      $('multilingual-evidence').replaceChildren(...cards.flatMap((lines, card) => lines.map((value, index) => {
        const p = document.createElement('p'); p.textContent = value;
        if (index === 0) p.className = card ? 'evidence-title evidence-next' : 'evidence-title';
        return p;
      })));
    }
    const limit = modeLimits();
    $('upload-help').textContent = `WAV, MP3, M4A & more · up to ${limitText(limit)}`;
    $('mode-footnote').textContent = conversation
      ? 'For people who take turns. Voices that talk at the same time are separated only with Speakers: "2 · at the same time" (clips up to 30 s, a demonstration), and speaker labels are not identities. Listen back to check the words.'
      : state.mode === 'separate'
        ? 'Designed for two overlapping speakers, English only. Source labels apply to this clip. Listen back to check the words.'
        : 'One English recording, no separation and no speaker labels. Listen back to check the words.';
    const demo = currentDemo();
    elements.demo.title = demo?.description || '';
    $('demo-title').textContent = conversation && state.twoVoices ? 'Kazakh + Russian: overlapping voices' : conversation ? 'Three languages. Six voices.' : 'Two voices. One recording.';
    $('demo-description').textContent = demo
      ? (conversation && state.twoVoices ? 'Controlled demonstration, not an accuracy benchmark.' : conversation ? 'Kazakh, Russian and English, one after another · assembled demo' : 'Two audiobook voices + noise · controlled demo')
      : 'Demo unavailable for this mode - upload your own recording';
  }

  function syncControls() {
    const problem = selectionProblem();
    if (problem && !state.busy) reportError(problem);
    const hasInput = Boolean(state.file || (state.useDemo && currentDemo()));
    elements.run.disabled = state.busy || !hasInput || Boolean(problem) || !readyForMode();
    elements.file.disabled = state.busy;
    elements.drop.disabled = state.busy;
    elements.demo.disabled = state.busy || !currentDemo();
    $('clear-file').disabled = state.busy;
    elements.export.disabled = state.busy || !state.result;
    $('delete-result').disabled = state.busy || !state.jobId;
    $('model-select').disabled = state.busy || state.locked;
    $('record-button').disabled = state.busy && !recorder;
    const conversationResult = state.result?.mode === 'conversation';
    show($('export-format'), Boolean(conversationResult));
    $('export-format').disabled = state.busy;
    syncConversationOptions();
    document.querySelectorAll('[data-mode]').forEach(button => {
      button.disabled = state.busy;
      const selected = button.dataset.mode === state.mode;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    $('run-label').textContent = state.busy ? 'Processing your recording…' : state.mode === 'conversation' ? 'Transcribe conversation' : state.mode === 'separate' ? 'Separate + transcribe' : 'Transcribe recording';
    $('run-caption').textContent = state.busy
      ? 'Your recording is being processed by the speech models.'
      : state.locked ? 'Enter the workspace access key above.'
        : !readyForMode() ? 'The models for this mode are not installed. Choose another mode or complete setup.'
        : hasInput ? 'Processed in this workspace. Results are private to this browser session.'
          : 'Select a file or load the demo to begin.';
    syncRecovery();
  }

  function clearSelection() {
    state.file = null;
    state.fileSeconds = null;
    state.useDemo = false;
    elements.file.value = '';
    show(elements.selected, false);
    elements.demo.classList.remove('demo-selected');
    reportError('');
    syncControls();
  }

  function setSelection(name, detail) {
    $('file-name').textContent = name;
    $('file-meta').textContent = detail;
    show(elements.selected, true);
    reportError('');
    syncControls();
  }

  async function selectFile(file) {
    if (!file || state.busy) return;
    const extension = file.name.split('.').pop().toLowerCase();
    if (!['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm', 'aac'].includes(extension)) {
      reportError('Choose an audio file such as WAV, MP3, M4A or FLAC. Video is not supported.'); return;
    }
    state.file = file;
    state.fileSeconds = null;
    state.useDemo = false;
    setSelection(file.name, `${(file.size / 1024 / 1024).toFixed(2)} MB · selected audio`);
    const fileGeneration = state.generation;
    const probe = document.createElement('audio');
    const url = URL.createObjectURL(file);
    probe.preload = 'metadata';
    let finished = false;
    const finish = () => { if (finished) return; finished = true; URL.revokeObjectURL(url); probe.removeAttribute('src'); probe.load(); };
    probe.onloadedmetadata = () => {
      if (state.file === file && fileGeneration === state.generation && Number.isFinite(probe.duration)) {
        state.fileSeconds = probe.duration;
        $('file-meta').textContent = `${probe.duration > 120 ? clock(probe.duration) : seconds(probe.duration)} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        syncControls();
      }
      finish();
    };
    probe.onerror = finish;
    probe.src = url;
    setTimeout(finish, 5000);
  }

  function showProgress(job) {
    const value = Math.max(0, Math.min(100, Number(job.progress) || 0));
    const rawStage = String(job.stage || 'Preparing audio');
    const readableStage = rawStage.replace(/[_-]+/g, ' ');
    $('progress-title').textContent = readableStage.charAt(0).toUpperCase() + readableStage.slice(1);
    $('progress-stage').textContent = job.status === 'queued' ? 'Waiting for the engine' : 'Running speech models';
    $('progress-fill').style.width = `${value}%`;
    $('progress-value').textContent = `${Math.round(value)}%`;
    $('progress-track').setAttribute('aria-valuenow', String(value));
  }

  function wait(milliseconds) { return new Promise(resolve => setTimeout(resolve, milliseconds)); }

  async function monitorJob(id, generation) {
    let failures = 0;
    while (generation === state.generation) {
      let job;
      try { job = await request(`/api/jobs/${encodeURIComponent(id)}`); failures = 0; }
      catch (error) {
        failures += 1;
        if (failures >= 4) throw error;
        $('progress-description').textContent = 'Reconnecting to the speech engine…';
        await wait(2000);
        continue;
      }
      if (generation !== state.generation) return;
      state.jobStatus = job.status;
      showProgress(job);
      $('progress-description').textContent = 'The first run may take longer while the models load.';
      if (job.status === 'error') throw new Error(job.error || 'Processing did not complete. Please try a different recording.');
      if (job.status === 'done') {
        if (!job.result) throw new Error('Processing finished without a result. Please try again.');
        state.result = job.result;
        try { state.speakerNames = JSON.parse(localStorage.getItem(`cleartalk-names-${id}`) || '{}'); } catch { state.speakerNames = {}; }
        rememberJob(id, job.status);
        state.mode = ['transcribe', 'conversation'].includes(job.result.mode) ? job.result.mode : 'separate';
        if (state.mode === 'conversation') {
          state.conversationModel = typeof job.result.model === 'string' ? job.result.model : 'auto';
          state.language = typeof job.result.language === 'string' ? job.result.language : 'auto';
          state.twoVoices = (job.result.requested_two_voices ?? job.result.two_voices) === true;
          if (state.twoVoices) state.speakers = 2;
        } else {
          state.model = job.result.model === 'cleartalk' ? 'cleartalk' : 'baseline';
        }
        $('model-select').value = state.model;
        updateModelDescription();
        renderResult(job.result, generation);
        return;
      }
      await wait(1200);
    }
  }

  async function run() {
    if (state.busy || (!state.file && !state.useDemo)) return;
    const generation = ++state.generation;
    state.busy = true;
    state.result = null;
    state.speakerNames = {}; state.edits = {}; state.editing = false;
    rememberJob(null, null);
    reportError('');
    show(elements.empty, false);
    show(elements.results, false);
    show(elements.progress, true);
    showProgress({ stage: 'Preparing your audio', progress: 0, status: 'queued' });
    document.querySelectorAll('audio').forEach(audio => audio.pause());
    syncControls();
    if (window.innerWidth < 760) elements.progress.scrollIntoView({ behavior: 'smooth', block: 'center' });
    try {
      const body = new FormData();
      body.append('mode', state.mode);
      if (state.mode === 'conversation') {
        body.append('model', state.conversationModel || 'auto');
        body.append('language', state.language || 'auto');
        body.append('speakers', String(state.speakers || 0));
        if (state.twoVoices) body.append('two_voices', 'true');
      } else {
        body.append('model', state.model);
      }
      if (state.useDemo) body.append('demo', 'true');
      else body.append('file', state.file);
      const created = await request('/api/jobs', { method: 'POST', body });
      if (!created.id) throw new Error('The server could not start a processing job. Please try again.');
      rememberJob(created.id, created.status || 'unknown');
      await monitorJob(created.id, generation);
    } catch (error) {
      if (generation !== state.generation) return;
      reportError(error.message);
      show(elements.progress, false);
      show(elements.empty, true);
    } finally {
      if (generation === state.generation) { state.busy = false; syncControls(); }
    }
  }

  async function recoverRecording() {
    if (!state.jobId || state.busy) return;
    const id = state.jobId;
    const generation = ++state.generation;
    state.busy = true; reportError(''); syncControls();
    show(elements.empty, false); show(elements.progress, true);
    try { await monitorJob(id, generation); }
    catch (error) {
      if (generation !== state.generation) return;
      reportError(error.message); show(elements.progress, false); show(elements.empty, true);
    } finally {
      if (generation === state.generation) { state.busy = false; syncControls(); }
    }
  }

  function transcript(element, text) {
    element.textContent = text?.trim() || 'No speech was transcribed for this track.';
    element.classList.toggle('unavailable', !text?.trim());
  }

  function makeSource(source, index, generation) {
    const article = document.createElement('article');
    article.className = 'audio-card source-card';
    const heading = document.createElement('div');
    heading.className = 'audio-card-heading';
    const headingGroup = document.createElement('div');
    const badge = document.createElement('span');
    badge.className = 'track-badge';
    badge.textContent = String.fromCharCode(65 + index);
    const nameGroup = document.createElement('div');
    const title = document.createElement('h3');
    title.textContent = source.label || `Source ${String.fromCharCode(65 + index)}`;
    const subtitle = document.createElement('p');
    subtitle.textContent = 'Estimated speech source';
    nameGroup.append(title, subtitle);
    headingGroup.append(badge, nameGroup);
    heading.append(headingGroup);
    const canvas = document.createElement('canvas');
    canvas.width = 700; canvas.height = 72; canvas.className = 'audio-waveform';
    canvas.setAttribute('aria-label', `Waveform of ${title.textContent}`);
    const audio = document.createElement('audio');
    audio.controls = true; audio.preload = 'metadata'; audio.src = source.audio_url;
    audio.setAttribute('aria-label', `Listen to ${title.textContent}`);
    const label = document.createElement('div');
    label.className = 'transcript-label'; label.textContent = 'TRANSCRIPT';
    const text = document.createElement('div'); text.className = 'transcript';
    transcript(text, source.text);
    article.append(heading, canvas, audio, label, text);
    if (Array.isArray(source.review_notes) && source.review_notes.length) {
      const note = document.createElement('p'); note.className = 'review-note'; note.textContent = source.review_notes.join(' '); article.append(note);
    }
    drawWaveform(source.audio_url, canvas, index ? '#89a275' : '#c5946c', generation);
    return article;
  }

  /* Conversation mode: one row per turn - time (click to listen), speaker, language, text.
     The user can name the speakers and correct text and language; exports use the corrected version. */
  const speakerName = (label) => state.speakerNames[label] || label;

  function currentTurns() {
    const turns = Array.isArray(state.result?.turns) ? state.result.turns : [];
    return turns.map((turn, index) => {
      const edit = state.edits[index] || {};
      const language = edit.language || turn.language;
      return { ...turn, ...edit, language, language_name: edit.language ? LANGUAGE_LABELS[edit.language] : (turn.language_name || LANGUAGE_LABELS[turn.language] || turn.language),
        name: speakerName(String(turn.speaker || 'Speaker')), edited: Boolean(state.edits[index]) };
    });
  }

  function renameSpeaker(label) {
    const current = speakerName(label);
    const name = window.prompt(`Name for ${label} (leave empty to go back to the label)`, current === label ? '' : current);
    if (name === null) return;
    const clean = name.trim().slice(0, 60);
    if (clean) state.speakerNames[label] = clean; else delete state.speakerNames[label];
    try { localStorage.setItem(`cleartalk-names-${state.jobId}`, JSON.stringify(state.speakerNames)); } catch { /* the names stay for this page only */ }
    renderConversation(state.result);
  }

  function renderConversation(result) {
    const turns = currentTurns();
    const labels = [...new Set(turns.map(turn => String(turn.speaker || 'Speaker')))];
    const color = label => SPEAKER_COLORS[Math.max(0, labels.indexOf(label)) % SPEAKER_COLORS.length];
    const speakers = Array.isArray(result.speakers) ? result.speakers : [];
    $('speaker-chips').replaceChildren(...speakers.map(speaker => {
      const chip = document.createElement('button'); chip.type = 'button'; chip.className = 'speaker-chip';
      chip.title = 'Give this speaker a name';
      const dot = document.createElement('span'); dot.className = 'speaker-dot'; dot.style.setProperty('--speaker', color(String(speaker.label)));
      const text = document.createElement('span');
      text.textContent = `${speakerName(String(speaker.label))} · ${clock(speaker.seconds)} · ${(speaker.languages || []).join(', ')}`;
      chip.append(dot, text);
      chip.addEventListener('click', () => renameSpeaker(String(speaker.label)));
      return chip;
    }));
    const languages = (Array.isArray(result.languages) ? result.languages : []).map(item => `${item.name} ${clock(item.seconds)}`);
    $('conversation-summary').textContent = `${speakers.length} speaker${speakers.length === 1 ? '' : 's'} · ${turns.length} turn${turns.length === 1 ? '' : 's'}${languages.length ? ` · ${languages.join(', ')}` : ''}`;
    const rows = turns.map((turn, index) => {
      const row = document.createElement('div'); row.className = `turn${turn.edited ? ' edited' : ''}`; row.style.setProperty('--speaker', color(String(turn.speaker)));
      row.dataset.start = String(turn.start); row.dataset.end = String(turn.end);
      const time = document.createElement('button'); time.type = 'button'; time.className = 'turn-time';
      time.textContent = `${clock(turn.start)} - ${clock(turn.end)}`;
      time.setAttribute('aria-label', `Play from ${clock(turn.start)}`);
      time.addEventListener('click', () => { const audio = $('original-audio'); audio.currentTime = Number(turn.start) || 0; audio.play().catch(() => {}); });
      const body = document.createElement('div');
      const meta = document.createElement('div'); meta.className = 'turn-meta';
      const who = document.createElement('span'); who.className = 'turn-speaker'; who.textContent = turn.name;
      if (state.speakerNames[String(turn.speaker)]) who.setAttribute('translate', 'no');
      meta.append(who);
      if (state.editing) {
        const pick = document.createElement('select'); pick.setAttribute('aria-label', 'Language of this turn');
        ['kk', 'ru', 'en', 'mixed'].forEach(code => { const option = document.createElement('option'); option.value = code; option.textContent = LANGUAGE_LABELS[code]; pick.append(option); });
        pick.value = LANGUAGE_LABELS[turn.language] ? turn.language : 'mixed';
        pick.addEventListener('change', () => { state.edits[index] = { ...(state.edits[index] || {}), language: pick.value }; row.classList.add('edited'); syncEditing(); });
        meta.append(pick);
      } else {
        const language = document.createElement('span'); language.className = 'turn-language'; language.textContent = String(turn.language_name || turn.language || '').toUpperCase();
        meta.append(language);
      }
      if (turn.overlap) { const flag = document.createElement('span'); flag.className = 'turn-flag'; flag.textContent = 'AT THE SAME TIME'; flag.title = 'Two people spoke at the same time here; their voices were separated before recognition.'; meta.append(flag); }
      if (turn.needs_review && !turn.edited) { const flag = document.createElement('span'); flag.className = 'turn-flag'; flag.textContent = 'CHECK'; flag.title = 'A recognition or language score crossed a review threshold.'; meta.append(flag); }
      const text = document.createElement('div'); text.className = 'turn-text'; text.textContent = turn.text || '';
      if (state.editing) {
        text.contentEditable = 'true'; text.spellcheck = false; text.setAttribute('translate', 'no');
        text.addEventListener('input', () => { state.edits[index] = { ...(state.edits[index] || {}), text: text.textContent }; row.classList.add('edited'); syncEditing(); });
      }
      body.append(meta, text); row.append(time, body); return row;
    });
    if (!rows.length) { const empty = document.createElement('div'); empty.className = 'turn-empty'; empty.textContent = 'No speech was transcribed in this recording.'; rows.push(empty); }
    $('turn-list').replaceChildren(...rows);
    const hidden = Array.isArray(result.background_turns) ? result.background_turns : [];
    show($('background-turns'), hidden.length > 0);
    $('background-summary').textContent = `Left out as background: ${hidden.length}`;
    $('background-list').replaceChildren(...hidden.map(turn => {
      const row = document.createElement('div'); row.className = 'background-turn';
      const time = document.createElement('span'); time.textContent = `${clock(turn.start)} - ${clock(turn.end)}`;
      const text = document.createElement('span'); text.textContent = turn.text || ''; text.setAttribute('translate', 'no');
      row.append(time, text); return row;
    }));
    syncEditing();
  }

  function syncEditing() {
    const changed = Object.keys(state.edits).length > 0;
    $('edit-turns').textContent = state.editing ? 'Finish correcting' : 'Correct transcript';
    show($('save-example'), state.editing || changed);
    $('turn-hint').textContent = state.editing
      ? 'Fix the words in each turn and its language. Then keep the recording as a check example, or just export the corrected text.'
      : 'Click a speaker to give a name. Click a time to listen from there.';
  }

  async function saveExample() {
    if (!state.jobId || state.result?.mode !== 'conversation') return;
    if (!window.confirm('Keep this recording and its corrected transcript on this computer as a check example? Do this only if everyone in the recording agreed.')) return;
    const turns = currentTurns().filter(turn => String(turn.text || '').trim()).map(turn => ({ start: Number(turn.start) || 0, end: Number(turn.end) || 0, speaker: turn.name, language: LANGUAGE_LABELS[turn.language] ? turn.language : 'mixed', text: String(turn.text).trim() }));
    $('save-example').disabled = true;
    try {
      const saved = await request(`/api/jobs/${encodeURIComponent(state.jobId)}/example`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turns, consent: true }) });
      $('example-status').textContent = `Saved. Check examples on this computer: ${saved.examples}.`;
    } catch (error) { $('example-status').textContent = error.message; }
    finally { $('save-example').disabled = false; }
  }

  /* Exports of a conversation are made here, so they carry the names and corrections. */
  function subtitleStamp(value, comma) {
    const ms = Math.max(0, Math.round((Number(value) || 0) * 1000));
    const pad = (number, width = 2) => String(number).padStart(width, '0');
    return `${pad(Math.floor(ms / 3600000))}:${pad(Math.floor(ms % 3600000 / 60000))}:${pad(Math.floor(ms % 60000 / 1000))}${comma ? ',' : '.'}${pad(ms % 1000, 3)}`;
  }

  function subtitleCues(turns) {
    // A long turn is cut into readable pieces; without word times its span is shared by text length.
    const cues = [];
    turns.forEach(turn => {
      const words = String(turn.text || '').split(/\s+/).filter(Boolean);
      if (!words.length) return;
      const pieces = []; let line = [];
      words.forEach(word => { if (line.length && line.join(' ').length + word.length + 1 > 84) { pieces.push(line.join(' ')); line = []; } line.push(word); });
      pieces.push(line.join(' '));
      const total = pieces.reduce((sum, piece) => sum + piece.length, 0);
      let cursor = Number(turn.start) || 0;
      const span = Math.max(0.5, (Number(turn.end) || 0) - cursor);
      pieces.forEach((piece, index) => { const length = span * piece.length / total; cues.push({ start: cursor, end: cursor + length, text: `${index === 0 ? `${turn.name}: ` : ''}${piece}` }); cursor += length; });
    });
    return cues;
  }

  function conversationFile(format) {
    const turns = currentTurns().map(turn => ({ ...turn, name: state.speakerNames[String(turn.speaker)] ? turn.name : inLanguage(turn.name) }));
    if (format === 'srt') return subtitleCues(turns).map((cue, index) => `${index + 1}\n${subtitleStamp(cue.start, true)} --> ${subtitleStamp(cue.end, true)}\n${cue.text}\n`).join('\n');
    if (format === 'vtt') return `WEBVTT\n\n${subtitleCues(turns).map(cue => `${subtitleStamp(cue.start)} --> ${subtitleStamp(cue.end)}\n${cue.text}\n`).join('\n')}`;
    const languageName = turn => String(turn.language_name || turn.language).split(' + ').map(inLanguage).join(' + ');
    const lines = [inLanguage('CLEARTALK - TRANSCRIPT'), inLanguage('Machine-generated text. Verify against the audio.'), '',
      ...turns.map(turn => `[${clock(turn.start)}-${clock(turn.end)}] ${turn.name} (${languageName(turn)}): ${turn.text}${turn.needs_review && !turn.edited ? `  ${inLanguage('[check]')}` : ''}`)];
    if (state.result.warnings?.length) lines.push('', inLanguage('LIMITATIONS'), ...state.result.warnings.map(warning => `• ${inLanguage(warning)}`));
    return `${lines.join('\n')}\n`;
  }

  /* Recording from the microphone: the result is handled exactly like a chosen file. */
  let recorder = null;
  async function toggleRecording() {
    if (recorder) { recorder.stop(); return; }
    if (state.busy) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { reportError('This browser cannot record audio. Upload a file instead.'); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch { reportError('The microphone is not available. Allow it in the browser, or upload a file.'); return; }
    const chunks = [];
    const started = Date.now();
    recorder = new MediaRecorder(stream);
    const timer = setInterval(() => {
      const seconds = (Date.now() - started) / 1000;
      $('record-label').textContent = `Stop recording · ${clock(seconds)}`;
      if (recorder && seconds >= modeLimits().seconds) recorder.stop();
    }, 500);
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearInterval(timer);
      stream.getTracks().forEach(track => track.stop());
      const seconds = (Date.now() - started) / 1000;
      const type = recorder.mimeType || 'audio/webm';
      recorder = null;
      $('record-button').classList.remove('recording');
      $('record-label').textContent = 'Record from the microphone';
      const extension = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm';
      const file = new File(chunks, `recording.${extension}`, { type });
      selectFile(file).then(() => {
        if (state.file !== file) return;
        state.fileSeconds = seconds;
        $('file-meta').textContent = `${seconds > 120 ? clock(seconds) : seconds.toFixed(1) + 's'} · recorded here`;
        syncControls();
      });
    };
    recorder.start(1000);
    reportError('');
    $('record-button').classList.add('recording');
    $('record-label').textContent = 'Stop recording · 0:00';
  }

  function renderResult(result, generation) {
    show(elements.progress, false);
    show(elements.results, true);
    const conversation = result.mode === 'conversation';
    const length = conversation ? clock(result.duration_seconds) : seconds(result.duration_seconds);
    $('result-duration').textContent = `${length} audio${Number.isFinite(Number(result.processing_seconds)) ? ` · ${conversation ? clock(result.processing_seconds) : seconds(result.processing_seconds)} processing` : ''}`;
    $('original-length').textContent = length;
    const original = result.original || {};
    $('original-audio').src = original.audio_url || '';
    transcript($('original-transcript'), original.text);
    show($('original-transcript'), !conversation);
    show($('original-transcript-label'), !conversation);
    show($('conversation-view'), conversation);
    if (conversation) { state.edits = {}; state.editing = false; $('example-status').textContent = ''; renderConversation(result); }
    // Decoding a long recording only to draw it would freeze the page: the waveform is for short clips.
    show($('original-waveform'), !(conversation && Number(result.duration_seconds) > 600));
    if (original.audio_url && !(conversation && Number(result.duration_seconds) > 600)) drawWaveform(original.audio_url, $('original-waveform'), '#8a9f81', generation);
    const sources = Array.isArray(result.sources) ? result.sources : [];
    $('source-grid').replaceChildren(...sources.map((source, index) => makeSource(source, index, generation)));
    show($('sources-heading'), sources.length > 0);
    show($('source-grid'), sources.length > 0);
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    $('warnings-list').replaceChildren(...warnings.map(warning => { const p = document.createElement('p'); p.textContent = warning; return p; }));
    $('warnings-summary').textContent = `Review notes${warnings.length ? ` (${warnings.length})` : ''}`;
    show($('result-warnings'), warnings.length > 0);
    const metrics = Array.isArray(result.metrics) ? result.metrics.filter(metric => Number.isFinite(Number(metric.value))) : [];
    $('result-metrics').replaceChildren(...metrics.map(metric => {
      const card = document.createElement('div'); card.className = 'metric-card';
      const value = document.createElement('strong'); value.textContent = `${Number(metric.value) > 0 ? '+' : ''}${Number(metric.value).toFixed(2)} ${metric.unit || ''}`.trim();
      const text = document.createElement('div');
      const label = document.createElement('span'); label.textContent = metric.label || 'Demo comparison';
      const explanation = document.createElement('p'); explanation.textContent = metric.description || 'Measured against the included demo reference recordings.';
      text.append(label, explanation); card.append(value, text); return card;
    }));
    show($('result-metrics'), metrics.length > 0);
    const modelNames = [result.mode === 'separate' && displayName(result.models?.separator), displayName(result.models?.asr), conversation && result.models?.speakers && `speaker labels from ${displayName(result.models.speakers)}`].filter(Boolean);
    $('result-methods').textContent = `${modelNames.length ? `Processed with ${modelNames.join(' + ')}. ` : ''}An unflagged segment may still contain errors.`;
  }

  async function drawWaveform(url, canvas, color, generation) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = color; ctx.globalAlpha = .35; ctx.beginPath(); ctx.moveTo(0, canvas.height / 2); ctx.lineTo(canvas.width, canvas.height / 2); ctx.stroke(); ctx.globalAlpha = 1;
    let audioContext;
    try {
      const response = await fetch(url);
      if (!response.ok) return;
      const buffer = await response.arrayBuffer();
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      audioContext = new AudioContextClass();
      const audio = await audioContext.decodeAudioData(buffer);
      if (generation !== state.generation) return;
      const samples = audio.getChannelData(0);
      const bars = 150;
      const step = Math.max(1, Math.floor(samples.length / bars));
      const peaks = [];
      let largest = .001;
      for (let i = 0; i < bars; i += 1) {
        let peak = 0;
        for (let j = i * step; j < Math.min((i + 1) * step, samples.length); j += Math.max(1, Math.floor(step / 80))) peak = Math.max(peak, Math.abs(samples[j]));
        peaks.push(peak); largest = Math.max(largest, peak);
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = color;
      const spacing = canvas.width / bars;
      peaks.forEach((peak, index) => { const height = Math.max(2, peak / largest * (canvas.height - 12)); ctx.fillRect(index * spacing, (canvas.height - height) / 2, Math.max(1.5, spacing * .42), height); });
    } catch { /* Audio controls remain usable if waveform decoding is unsupported. */ }
    finally { if (audioContext) audioContext.close().catch(() => {}); }
  }

  function saveFile(text, name, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportText() {
    if (!state.result) return;
    if (state.result.mode === 'conversation') {
      const format = ['srt', 'vtt'].includes($('export-format').value) ? $('export-format').value : 'txt';
      saveFile(conversationFile(format), `cleartalk-transcript.${format}`, format === 'vtt' ? 'text/vtt;charset=utf-8' : 'text/plain;charset=utf-8');
      return;
    }
    // The server keeps an English text file; in another interface language the page writes its own.
    if (state.result.download_url && (window.ClearTalkLanguage?.current || 'en') === 'en') {
      const link = document.createElement('a'); link.href = state.result.download_url; link.download = 'cleartalk-transcript.txt'; link.click(); return;
    }
    const result = state.result;
    const parts = ['CLEARTALK TRANSCRIPT', 'Machine-generated text. Verify against the audio.', '', 'ORIGINAL RECORDING'].map(inLanguage);
    parts.push(result.original?.text || inLanguage('(No speech transcribed)'), ...((result.sources || []).flatMap(source => ['', inLanguage(source.label || 'SOURCE'), source.text || inLanguage('(No speech transcribed)')])));
    if (result.warnings?.length) parts.push('', inLanguage('REVIEW NOTES'), ...result.warnings.map(inLanguage));
    saveFile(parts.join('\n'), 'cleartalk-transcript.txt', 'text/plain;charset=utf-8');
  }

  elements.drop.addEventListener('click', () => elements.file.click());
  elements.file.addEventListener('change', () => selectFile(elements.file.files[0]));
  ['dragenter', 'dragover'].forEach(event => elements.drop.addEventListener(event, e => { e.preventDefault(); if (!state.busy) elements.drop.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(event => elements.drop.addEventListener(event, e => { e.preventDefault(); elements.drop.classList.remove('drag-over'); }));
  elements.drop.addEventListener('drop', e => { if (!state.busy) selectFile(e.dataTransfer.files[0]); });
  elements.demo.addEventListener('click', () => {
    const demo = currentDemo();
    if (!demo || state.busy) return;
    state.file = null; state.fileSeconds = null; state.useDemo = true; elements.file.value = '';
    setSelection(demo.title || 'Included demo', state.mode === 'conversation' && !state.twoVoices ? 'Demo loaded · three languages, six readers' : 'Demo loaded · two overlapping voices');
  });
  $('clear-file').addEventListener('click', clearSelection);
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    const before = currentDemo();
    state.mode = button.dataset.mode;
    if (state.useDemo && currentDemo() !== before) { state.useDemo = false; show(elements.selected, false); }
    reportError(''); syncControls();
  }));
  $('language-select').addEventListener('change', () => { state.language = $('language-select').value; syncControls(); });
  $('speakers-select').addEventListener('change', () => {
    const before = currentDemo();
    const value = $('speakers-select').value;
    state.twoVoices = value === 'two-voices';
    state.speakers = state.twoVoices ? 2 : Number(value) || 0;
    if (state.useDemo && currentDemo() !== before) { state.useDemo = false; show(elements.selected, false); }
    syncControls();
  });
  $('conversation-model-select').addEventListener('change', () => { state.conversationModel = $('conversation-model-select').value; syncControls(); });
  elements.run.addEventListener('click', run);
  elements.export.addEventListener('click', exportText);
  $('record-button').addEventListener('click', toggleRecording);
  $('edit-turns').addEventListener('click', () => { state.editing = !state.editing; if (state.result) renderConversation(state.result); });
  $('save-example').addEventListener('click', saveExample);
  async function deleteRecording() {
    if (!state.jobId || state.busy || !['done', 'error'].includes(state.jobStatus) || !window.confirm('Permanently delete this result, its audio and transcript from this workspace? Download anything you want to keep first.')) return;
    state.busy = true; syncControls();
    try {
      await request(`/api/jobs/${encodeURIComponent(state.jobId)}`, { method: 'DELETE' });
      document.querySelectorAll('audio').forEach(audio => { audio.pause(); audio.removeAttribute('src'); audio.load(); });
      rememberJob(null, null); state.result = null; state.generation += 1;
      show(elements.results, false); show(elements.empty, true);
      notice('The processed result, audio and transcript were deleted from this workspace. This cannot be undone. Your original file on your device was not changed.');
    } catch (error) { reportError(error.message); }
    finally { state.busy = false; syncControls(); }
  }
  $('delete-result').addEventListener('click', deleteRecording);
  checkRecording.addEventListener('click', recoverRecording);
  deleteRecordingButton.addEventListener('click', deleteRecording);
  $('model-select').addEventListener('change', () => { state.model = $('model-select').value; updateModelDescription(); syncControls(); });
  $('original-audio').addEventListener('timeupdate', () => {
    const now = $('original-audio').currentTime;
    document.querySelectorAll('.turn').forEach(row => row.classList.toggle('playing', now >= Number(row.dataset.start) && now < Number(row.dataset.end)));
  });
  $('access-panel').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = $('access-panel').querySelector('button');
    button.disabled = true; $('access-error').textContent = '';
    try {
      await request('/api/access', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: $('access-key').value }) });
      $('access-key').value = ''; state.locked = false; show($('access-panel'), false); notice('');
      await initialize();
    } catch (error) { $('access-error').textContent = error.message; }
    finally { button.disabled = false; }
  });
  const dialog = $('guide-dialog');
  ['guide-button', 'scope-button'].forEach(id => $(id).addEventListener('click', () => dialog.showModal()));
  $('close-guide').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', e => { if (e.target === dialog) { const rect = dialog.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) dialog.close(); } });
  document.addEventListener('play', event => { if (event.target.tagName === 'AUDIO') document.querySelectorAll('audio').forEach(audio => { if (audio !== event.target) audio.pause(); }); }, true);
  window.addEventListener('beforeunload', event => { if (state.busy) { event.preventDefault(); event.returnValue = ''; } });

  function updateModelDescription() {
    const models = state.health?.models || {};
    const ours = models.cleartalk;
    const card = ours?.model_card || {};
    $('model-description').textContent = state.model === 'cleartalk'
      ? 'Whisper small.en adapted in this project (LoRA) to separated, noisy speech. Compare it with the pretrained model on your own recordings.'
      : 'Whisper small.en as released by OpenAI, without ClearTalk training.';
    show($('training-details'), Boolean(ours?.available));
    const percent = value => `${(100 * value).toFixed(1)}%`;
    const count = value => Number.isSafeInteger(value) && value >= 0;
    const rate = value => Number.isFinite(value) && value >= 0;
    const lines = [card.model_name || 'ClearTalk fine-tuned recognizer'];
    if (count(card.trainable_parameters) && count(card.total_parameters) && card.total_parameters > 0) {
      lines.push(`${card.trainable_parameters.toLocaleString('en-US')} trained LoRA parameters (${(100 * card.trainable_parameters / card.total_parameters).toFixed(1)}% of ${card.total_parameters.toLocaleString('en-US')}); the merged model has the same size and speed as the pretrained one.`);
    }
    const training = card.training || {};
    if (count(training.tracks) && count(training.speakers)) {
      lines.push(`Trained on ${training.tracks.toLocaleString('en-US')} separated LibriSpeech tracks from ${training.speakers} speakers; validation and test speakers were never seen in training.`);
    }
    const describe = (label, block, unit) => {
      if (!block || !rate(block.baseline_wer) || !rate(block.cleartalk_wer) || !count(block[unit])) return;
      lines.push(`${label}: word error rate ${percent(block.baseline_wer)} pretrained → ${percent(block.cleartalk_wer)} ClearTalk on the same ${block[unit]} ${unit}. Lower is better.`);
    };
    describe('Validation, two-speaker mixtures', card.validation, 'mixtures');
    describe('Validation, single clean speaker', card.validation_clean, 'utterances');
    describe('Final test, two-speaker mixtures', card.test, 'mixtures');
    describe('Final test, single clean speaker', card.test_clean, 'utterances');
    lines.push('Measured on synthetic mixtures of read English audiobooks. Real meetings, accents and microphones can behave differently.');
    $('training-evidence').replaceChildren(...lines.map(value => { const p = document.createElement('p'); p.textContent = value; return p; }));
  }

  async function initialize() {
    syncControls();
    try {
      const session = await request('/api/session');
      $('retention-note').textContent = `Uploaded audio is processed on the machine hosting this workspace. Results are available to this browser session and automatically expire after ${session.retention_hours || 24} hours; you can delete a finished result sooner. The server owner can access stored files. Only upload recordings you have permission to process.`;
      state.locked = Boolean(session.access_required && !session.authenticated);
      show($('access-panel'), state.locked);
      if (state.locked) { $('engine-status').lastChild.textContent = 'Workspace locked'; syncControls(); return; }
    } catch (error) { notice(error.message); }
    const outcomes = await Promise.allSettled([request('/api/health'), request('/api/demo'), request('/api/demo/conversation'), request('/api/demo/overlap')]);
    const health = outcomes[0];
    if (health.status === 'fulfilled') {
      state.health = health.value;
      $('cleartalk-model-option').disabled = !health.value.models?.cleartalk?.available;
      const untouched = !state.result && !state.file && !state.useDemo;
      if (untouched && health.value.default_model === 'cleartalk' && health.value.models?.cleartalk?.available) {
        state.model = 'cleartalk';
        $('model-select').value = state.model;
      }
      // Conversation is the general mode: it opens first when its models are installed.
      if (untouched && health.value.features?.conversation_mode && !new URLSearchParams(window.location.search).get('job')) state.mode = 'conversation';
      updateModelDescription();
      const ready = health.value.status === 'ready';
      $('engine-status').className = `engine-status ${ready ? 'ready' : 'warning'}`;
      $('engine-status').lastChild.textContent = ready ? 'Speech engine ready' : 'Model setup needed';
      if (!ready) notice(health.value.message || 'ClearTalk is running, but its speech models still need to be installed. Follow the setup instructions, then refresh this page.');
    } else {
      $('engine-status').className = 'engine-status warning';
      $('engine-status').lastChild.textContent = 'Engine unavailable';
      notice(health.reason.message);
    }
    const demo = outcomes[1];
    if (demo.status === 'fulfilled' && demo.value.audio_url) {
      state.demo = demo.value;
      if (demo.value.description) elements.demo.title = demo.value.description;
    }
    const conversationDemo = outcomes[2];
    if (conversationDemo.status === 'fulfilled' && conversationDemo.value.audio_url) state.conversationDemo = conversationDemo.value;
    if (outcomes[3].status === 'fulfilled') state.overlapDemo = outcomes[3].value;
    syncControls();
    const savedJob = new URLSearchParams(window.location.search).get('job');
    if (savedJob) {
      rememberJob(savedJob, 'unknown');
      const generation = ++state.generation;
      state.busy = true;
      show(elements.empty, false);
      show(elements.progress, true);
      showProgress({ stage: 'Opening your saved result', progress: 0, status: 'running' });
      syncControls();
      try { await monitorJob(savedJob, generation); }
      catch (error) {
        show(elements.progress, false);
        show(elements.empty, true);
        reportError(error.message);
      } finally { state.busy = false; syncControls(); }
    }
  }
  initialize();
})();
