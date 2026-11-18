/**
 * Speech Studio page.
 *
 * Builds the composer, validates input, calls the API and renders the
 * result. All speech logic stays in the Python backend (`app_core.py`).
 */

import { ApiError, generateSpeech } from "../api.js";
import { createAudioPanel } from "../components/audio-player.js";
import {
  describeVoice,
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
  const labels = engines.labels || {};
  const voices = engines.kokoro?.voices || [];
  const voiceLabels = engines.kokoro?.voice_labels || {};
  const maxLength = capabilities.limits?.max_text_length || DEFAULT_MAX_LENGTH;

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

  const selectors = document.createElement("div");
  selectors.className = "field-row";

  const engineField = document.createElement("div");
  engineField.className = "field";

  const engineSelect = document.createElement("select");
  engineSelect.className = "select";
  engineSelect.id = "engine-select";

  for (const [id, label] of Object.entries(labels)) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = label;
    engineSelect.append(option);
  }

  if (engines.default) {
    engineSelect.value = engines.default;
  }

  engineField.append(fieldLabel("Engine", "engine-select"), engineSelect);

  const voiceField = document.createElement("div");
  voiceField.className = "field";

  const voiceSelect = document.createElement("select");
  voiceSelect.className = "select";
  voiceSelect.id = "voice-select";

  for (const id of voices) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = voiceLabels[id] || id;
    voiceSelect.append(option);
  }

  voiceField.append(fieldLabel("Voice", "voice-select"), voiceSelect);

  selectors.append(engineField, voiceField);

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

  composerBody.append(textField, metaRow, selectors, examples, actions);
  composer.append(composerBody);

  // --- Output -----------------------------------------------------------
  const audioPanel = createAudioPanel();
  const side = document.createElement("div");
  side.className = "stack";
  side.append(audioPanel.element);

  grid.append(composer, side);

  // --- Behaviour --------------------------------------------------------
  function voiceIsRelevant() {
    return voiceControlsVisible(engineSelect.value);
  }

  function syncVoiceVisibility() {
    voiceField.hidden = !voiceIsRelevant();
    voiceSelect.disabled = !voiceIsRelevant();
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
    voiceSelect.disabled = busy || !voiceIsRelevant();
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

    setBusy(true);

    try {
      const result = await generateSpeech({
        text: validation.text,
        engine,
        voice,
      });

      audioPanel.show(result, {
        engineLabel: labels[result.engine] || result.engine,
        voiceLabel: describeVoice(result.engine, voice, voiceLabels),
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
    syncVoiceVisibility();
    setBusy(gate.isRunning);
  });

  syncVoiceVisibility();
  updateCounter();

  return { element: grid, startGeneration };
}
