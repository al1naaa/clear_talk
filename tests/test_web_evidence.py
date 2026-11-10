"""Run the actual evidence renderer against generated cards, without a browser."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def node():
    executable = os.environ.get('CLEARTALK_TEST_NODE') or shutil.which('node')
    if not executable:
        pytest.skip('Node.js is needed for the bounded web rendering checks')
    return executable


def render(node, card, model='cleartalk'):
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('web/app.js', 'utf8');
const start = source.indexOf('  function updateModelDescription() {');
const end = source.indexOf('  async function initialize()', start);
if (start < 0 || end < 0) throw new Error('Evidence renderer not found');
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, {textContent: '', replaceChildren(...children) {this.children = children;}});
  return elements.get(id);
};
const card = JSON.parse(process.argv[1]);
const context = {state: {model: process.argv[2], health: {models: {cleartalk: {available: true, model_card: card}}}},
  $: get, show() {}, document: {createElement() {return {textContent: ''};}}};
vm.runInNewContext(source.slice(start, end) + '\nupdateModelDescription();', context);
process.stdout.write(JSON.stringify({lines: get('training-evidence').children.map(row => row.textContent),
                                     description: get('model-description').textContent}));
"""
    result = subprocess.run([node, '-e', script, json.dumps(card), model], cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


FULL_CARD = {
    'model_name': 'ClearTalk Whisper small.en (LoRA fine-tuned)', 'trainable_parameters': 7077888,
    'total_parameters': 248842752, 'training': {'tracks': 17000, 'speakers': 249},
    'validation': {'baseline_wer': 0.365, 'cleartalk_wer': 0.30, 'mixtures': 348},
    'validation_clean': {'baseline_wer': 0.04, 'cleartalk_wer': 0.041, 'utterances': 300},
}


def test_javascript_syntax(node):
    subprocess.run([node, '--check', str(ROOT / 'web/app.js')], check=True, capture_output=True, text=True)


def test_full_card_renders_measured_numbers(node):
    out = render(node, FULL_CARD)
    text = ' '.join(out['lines'])
    assert '7,077,888 trained LoRA parameters (2.8% of 248,842,752)' in text
    assert '36.5% pretrained → 30.0% ClearTalk on the same 348 mixtures' in text
    assert '4.0% pretrained → 4.1% ClearTalk on the same 300 utterances' in text
    assert 'synthetic mixtures of read English audiobooks' in text
    assert 'Final test' not in text  # never invented before the one-time test exists
    assert 'LoRA' in out['description']


@pytest.mark.parametrize('broken', [{}, {'validation': {'baseline_wer': '0.3', 'cleartalk_wer': 0.2, 'mixtures': 5}},
                                    {'validation': {'baseline_wer': 0.3, 'cleartalk_wer': -1, 'mixtures': 5}},
                                    {'trainable_parameters': 5, 'total_parameters': 0}])
def test_missing_or_invalid_numbers_are_not_rendered(node, broken):
    text = ' '.join(render(node, broken)['lines'])
    assert 'undefined' not in text and 'NaN' not in text
    assert 'pretrained →' not in text and 'trained LoRA parameters' not in text


def test_baseline_description_does_not_claim_training(node):
    assert 'without ClearTalk training' in render(node, FULL_CARD, model='baseline')['description']


def test_guide_describes_current_model_and_limits():
    html = (ROOT / 'web/index.html').read_text()
    assert 'https://www.openslr.org/12/' in html
    assert 'controlled mixtures, not natural conversations' in html
    assert 'do not establish real-world accuracy' in html
    assert 'LoRA' in html and 'refiner' not in html
    assert '<option value="cleartalk" id="cleartalk-model-option" disabled>' in html


def test_interface_shows_no_version_label():
    html = (ROOT / 'web/index.html').read_text()
    source = (ROOT / 'web/app.js').read_text()
    assert 'profile-version' not in html and 'profile-version' not in source
    assert 'updateProfileVersion' not in source
    assert not re.search(r'\bV0\.\d|\bv0\.\d|version \d', html + source, flags=re.I)


@pytest.mark.parametrize('scenario', ['inference_error', 'poll_error', 'recover', 'active_delete', 'failed_delete'])
def test_recording_recovery_keeps_id_and_respects_active_worker(node, scenario):
    script = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync('web/app.js', 'utf8');
const slice = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
const functions = slice('  function rememberJob(', '  function reportError(')
  + slice('  async function monitorJob(', '  function transcript(')
  + slice('  async function deleteRecording()', "  $('delete-result').addEventListener");
const scenario = process.argv[1], id = 'a'.repeat(32);
const state = {file: {}, useDemo: false, mode: 'transcribe', model: 'baseline', generation: 0,
  busy: false, result: null, jobId: null, jobStatus: null};
const nodes = new Map(), get = key => {if (!nodes.has(key)) nodes.set(key, {}); return nodes.get(key);};
const recovery = {}, deleteButton = {}, checkButton = {}, recoveryText = {};
const window = {innerWidth: 1000, location: {href: 'http://localhost/'}, confirm: () => true,
  history: {replaceState(_a, _b, url) {window.location.href = String(url);}}};
let deletes = 0, recover = false;
const context = {state, URL, window, recovery, deleteRecordingButton: deleteButton,
  checkRecording: checkButton, recoveryText, $: get, elements: {empty:{}, results:{}, progress:{}},
  FormData: class {append() {}}, document: {querySelectorAll() {return [];}},
  show(element, visible) {element.visible = visible;}, showProgress() {}, reportError() {}, notice() {},
  updateModelDescription() {}, renderResult() {}, wait: async () => {},
  request: async (url, options) => {
    if (options?.method === 'POST') return {id, status:'queued'};
    if (options?.method === 'DELETE') {deletes++; return null;}
    assert.equal(state.jobId, id); // ID must already be retained before first polling request.
    assert.equal(new URL(window.location.href).searchParams.get('job'), id);
    if (recover) return {status:'done', result:{mode:'transcribe', model:'baseline'}};
    if (scenario === 'inference_error' || scenario === 'failed_delete') return {status:'error', error:'fixture failure'};
    throw new Error('fixture network outage');
  }};
vm.createContext(context);
vm.runInContext(functions + '\nfunction syncControls() {syncRecovery();}', context);
(async () => {
  await vm.runInContext('run()', context);
  assert.equal(state.jobId, id);
  assert.equal(state.busy, false);
  assert.equal(recovery.visible, true);
  const failed = scenario === 'inference_error' || scenario === 'failed_delete';
  assert.equal(deleteButton.disabled, !failed);
  if (scenario === 'recover') {
    recover = true;
    await vm.runInContext('recoverRecording()', context);
    assert.equal(state.jobStatus, 'done'); assert.ok(state.result); assert.equal(recovery.visible, false);
  }
  if (scenario === 'active_delete' || scenario === 'failed_delete') {
    await vm.runInContext('deleteRecording()', context);
    assert.equal(deletes, scenario === 'failed_delete' ? 1 : 0);
    assert.equal(state.jobId, scenario === 'failed_delete' ? null : id);
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    subprocess.run([node, '-e', script, scenario], cwd=ROOT, capture_output=True, text=True, check=True)
