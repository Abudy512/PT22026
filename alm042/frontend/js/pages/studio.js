/**
 * Speech Studio page.
 *
 * Builds the composer, validates input, calls the API and renders the
 * result. All speech logic stays in the Python backend (`app_core.py`), and
 * every control (engines, voices, speed range) comes from `/api/engines`.
 */

import { ApiError, generateSpeech } from "../api.js";
import { createAudioPanel } from "../components/audio-player.js";
import {
  describeSpeed,
  describeVoice,
  formatSpeed,
  speedControlVisible,
  speedForRequest,
  voiceControlsVisible,
  voiceForRequest,
} from "../engine-ui.js";
import { KeyAction, resolveKeyAction } from "../keyboard.js";
import { createGenerationGate } from "../state.js";
import {
  DEFAULT_MAX_LENGTH,
  countCharacters,
  counterState,
  validateText,
} from "../validation.js";

const EXAMPLES = [
  "Hello, welcome to VoiceForge.",
  "The quick brown fox jumps over the lazy dog.",
  "This is a test of the speech synthesis system.",
];

function fieldLabel(text, targetId) {
  const label = document.createElement("label");
  label.className = "field-label";
  label.setAttribute("for", targetId);
  label.textContent = text;
  return label;
}

/**
 * @param {{capabilities: object, notifications: object}} options
 */
export function createStudioPage({ capabilities, notifications }) {
  const engines = capabilities.engines || {};
  const defaultEngine = capabilities.default;
  const maxLength = capabilities.limits?.max_text_length || DEFAULT_MAX_LENGTH;

  const kokoro = engines.kokoro || {};
  const kokoroVoices = kokoro.voices || [];
  const speedRange = kokoro.speed || {};

  const voiceLabels = Object.fromEntries(
    kokoroVoices.map((voice) => [voice.id, voice.name]),
  );

  const gate = createGenerationGate();

  const grid = document.createElement("div");
  grid.className = "studio-grid";

  // --- Composer ---------------------------------------------------------
  const composer = document.createElement("section");
  composer.className = "panel";
  composer.innerHTML = `
    <div class="panel-head">
      <h2 class="panel-title">Compose</h2>
      <span class="panel-hint">English text</span>
    </div>
  `;

  const composerBody = document.createElement("div");
  composerBody.className = "panel-body stack";

  const textField = document.createElement("div");
  textField.className = "field";

  const textarea = document.createElement("textarea");
  textarea.className = "textarea";
  textarea.id = "speech-text";
  textarea.maxLength = maxLength;
  textarea.placeholder = "Type or paste a sentence, for example: Welcome to VoiceForge.";
  textarea.setAttribute("aria-describedby", "composer-hint");

  textField.append(fieldLabel("Text", "speech-text"), textarea);

  const metaRow = document.createElement("div");
  metaRow.className = "textarea-meta";

  const hint = document.createElement("span");
  hint.id = "composer-hint";
  hint.textContent = "Press Enter to generate \u00b7 Shift + Enter for a new line";

  const counter = document.createElement("span");
  counter.className = "counter";
  counter.textContent = `0 / ${maxLength}`;

  metaRow.append(hint, counter);

  // Engine + voice
  const selectors = document.createElement("div");
  selectors.className = "field-row";

  const engineField = document.createElement("div");
  engineField.className = "field";

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";
  engineSelect.id = "engine-select";

  for (const [id, info] of Object.entries(engines)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = info.name || id;
    engineSelect.append(option);
  }

  if (defaultEngine && engines[defaultEngine]) {
    engineSelect.value = defaultEngine;
  }

  engineField.append(fieldLabel("Engine", "engine-select"), engineSelect);

  const voiceField = document.createElement("div");
  voiceField.className = "field";

  const voiceSelect = document.createElement("select");
  voiceSelect.className = "select";
  voiceSelect.id = "voice-select";

  for (const voice of kokoroVoices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.name;
    voiceSelect.append(option);
  }

  const defaultVoice = kokoroVoices.find((voice) => voice.default);

  if (defaultVoice) {
    voiceSelect.value = defaultVoice.id;
  }

  voiceField.append(fieldLabel("Voice", "voice-select"), voiceSelect);

  selectors.append(engineField, voiceField);

  // Speed (Kokoro only)
  const speedField = document.createElement("div");
  speedField.className = "field";
  speedField.id = "speed-field";

  const speedLabelRow = document.createElement("div");
  speedLabelRow.className = "field-label-row";

  const speedLabel = document.createElement("label");
  speedLabel.className = "field-label";
  speedLabel.setAttribute("for", "speed-range");
  speedLabel.textContent = "Speed";

  const speedValue = document.createElement("span");
  speedValue.className = "badge badge-accent";
  speedValue.id = "speed-value";

  speedLabelRow.append(speedLabel, speedValue);

  const speedInput = document.createElement("input");
  speedInput.type = "range";
  speedInput.className = "range";
  speedInput.id = "speed-range";
  speedInput.min = speedRange.min ?? 0.5;
  speedInput.max = speedRange.max ?? 2;
  speedInput.step = "0.05";
  speedInput.value = speedRange.default ?? 1;

  const speedScale = document.createElement("div");
  speedScale.className = "range-scale";

  const slow = document.createElement("span");
  slow.textContent = `${speedInput.min}\u00d7 slower`;

  const fast = document.createElement("span");
  fast.textContent = `${speedInput.max}\u00d7 faster`;

  speedScale.append(slow, fast);
  speedField.append(speedLabelRow, speedInput, speedScale);

  const speedNote = document.createElement("p");
  speedNote.className = "field-note";
  speedNote.id = "speed-note";
  speedNote.hidden = true;
  speedNote.textContent =
    "Speed is only available for the Kokoro engine. " +
    "The VoiceForge custom model always uses its native rate.";

  // Actions
  const actions = document.createElement("div");
  actions.className = "composer-actions";

  const spinner = document.createElement("span");
  spinner.className = "spinner";
  spinner.hidden = true;

  const generateButton = document.createElement("button");
  generateButton.type = "button";
  generateButton.className = "button button-primary";
  generateButton.id = "generate-button";
  generateButton.append(spinner, document.createTextNode("Generate speech"));

  actions.append(generateButton);

  const examples = document.createElement("div");
  examples.className = "chip-row";

  for (const example of EXAMPLES) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = example;
    chip.addEventListener("click", () => {
      textarea.value = example;
      updateCounter();
      textarea.focus();
    });
    examples.append(chip);
  }

  composerBody.append(
    textField,
    metaRow,
    selectors,
    speedField,
    speedNote,
    examples,
    actions,
  );
  composer.append(composerBody);

  // --- Output -----------------------------------------------------------
  const audioPanel = createAudioPanel();
  const side = document.createElement("div");
  side.className = "stack";
  side.append(audioPanel.element);

  grid.append(composer, side);

  // --- Behaviour --------------------------------------------------------
  function syncVisibility() {
    const engineId = engineSelect.value;
    const voiceEnabled = voiceControlsVisible(engineId);
    const speedEnabled = speedControlVisible(engineId);

    voiceField.hidden = !voiceEnabled;
    voiceSelect.disabled = !voiceEnabled;
    speedField.hidden = !speedEnabled;
    speedInput.disabled = !speedEnabled;
    speedNote.hidden = speedEnabled;
  }

  function updateSpeedLabel() {
    speedValue.textContent = formatSpeed(speedInput.value);
  }

  function updateCounter() {
    const count = countCharacters(textarea.value);
    const state = counterState(count, maxLength);

    counter.textContent = `${count} / ${maxLength}`;
    counter.classList.toggle("is-warning", state === "warning");
    counter.classList.toggle("is-over", state === "over");
  }

  function setBusy(busy) {
    generateButton.disabled = busy;
    spinner.hidden = !busy;
    engineSelect.disabled = busy;
    voiceSelect.disabled = busy || !voiceControlsVisible(engineSelect.value);
    speedInput.disabled = busy || !speedControlVisible(engineSelect.value);
    textarea.readOnly = busy;
    generateButton.lastChild.textContent = busy
      ? "Generating\u2026"
      : "Generate speech";
  }

  /**
   * The single generation entry point. The button and the Enter key both
   * call this; the gate guarantees one request at a time.
   */
  async function startGeneration() {
    if (!gate.tryStart()) {
      return;
    }

    const validation = validateText(textarea.value, maxLength);

    if (!validation.ok) {
      gate.finish();
      notifications.error(validation.error);
      textarea.focus();
      return;
    }

    const engine = engineSelect.value;
    const voice = voiceForRequest(engine, voiceSelect.value);
    const speed = speedForRequest(engine, speedInput.value);

    setBusy(true);

    try {
      const result = await generateSpeech({
        text: validation.text,
        engine,
        voice,
        speed,
      });

      audioPanel.show(result, {
        engineLabel: engines[result.engine]?.name || result.engine,
        voiceLabel: describeVoice(result.engine, result.voice, voiceLabels),
        speedLabel: describeSpeed(result.engine, result.speed),
      });

      notifications.success(
        `Generated ${result.duration_seconds.toFixed(2)} s of audio.`,
      );
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : "Speech generation failed unexpectedly.";

      notifications.error(message);
    } finally {
      gate.finish();
      setBusy(false);
    }
  }

  textarea.addEventListener("input", updateCounter);

  speedInput.addEventListener("input", updateSpeedLabel);

  textarea.addEventListener("keydown", (event) => {
    const action = resolveKeyAction(event, textarea.value);

    // Shift+Enter and IME composition keep the browser's default behaviour.
    if (action === KeyAction.NEWLINE || event.isComposing) {
      return;
    }

    if (event.key !== "Enter") {
      return;
    }

    // Do not hijack Ctrl/Cmd/Alt+Enter.
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }

    // Plain Enter never inserts a newline in the composer.
    event.preventDefault();

    // Even when the text is empty we call the shared entry point: it
    // validates, explains the problem and stops.
    startGeneration();
  });

  generateButton.addEventListener("click", startGeneration);

  engineSelect.addEventListener("change", () => {
    syncVisibility();
    setBusy(gate.isRunning);
  });

  syncVisibility();
  updateSpeedLabel();
  updateCounter();

  return { element: grid, startGeneration };
}
