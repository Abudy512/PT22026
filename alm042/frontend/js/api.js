/**
 * Thin HTTP client for the VoiceForge backend.
 *
 * Every network call lives here so the UI code never deals with fetch(),
 * status codes or JSON parsing directly.
 */

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function requestJson(url, options = {}) {
  let response;

  try {
    response = await fetch(url, options);
  } catch (error) {
    if (error.name === "AbortError") {
      throw error;
    }

    throw new ApiError(
      "Cannot reach the VoiceForge server. Is it still running?",
      0,
    );
  }

  const payload = await readJson(response);

  if (!response.ok || !payload || payload.ok !== true) {
    const message =
      (payload && payload.error) ||
      `The server returned an error (HTTP ${response.status}).`;

    throw new ApiError(message, response.status);
  }

  return payload;
}

/** Engine capabilities, voices and text limits. */
export async function fetchCapabilities() {
  return requestJson("/health");
}

/**
 * Generate speech.
 *
 * @param {{text: string, engine: string, voice?: string|null, signal?: AbortSignal}} request
 */
export async function generateSpeech({ text, engine, voice, signal }) {
  const body = { text, engine };

  if (voice) {
    body.voice = voice;
  }

  return requestJson("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}
