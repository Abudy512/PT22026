/**
 * Engine-related UI rules, kept pure so they can be tested without a DOM.
 */

export const ENGINE_KOKORO = "kokoro";
export const ENGINE_VOICEFORGE = "voiceforge";

/** Only the Kokoro engine has selectable voices. */
export function voiceControlsVisible(engineId) {
  return engineId === ENGINE_KOKORO;
}

/** The voice is only sent to the backend for Kokoro. */
export function voiceForRequest(engineId, voiceId) {
  return engineId === ENGINE_KOKORO ? voiceId || null : null;
}

/** Human label for the voice shown in the output metadata. */
export function describeVoice(engineId, voiceId, voiceLabels = {}) {
  if (engineId !== ENGINE_KOKORO) {
    return "not applicable";
  }

  return voiceLabels[voiceId] || voiceId || "default";
}
