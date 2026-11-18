import test from "node:test";
import assert from "node:assert/strict";

import {
  describeVoice,
  voiceControlsVisible,
  voiceForRequest,
} from "../js/engine-ui.js";

const LABELS = { af_heart: "American female — warm and clear" };

test("the voice selector is visible only for Kokoro", () => {
  assert.equal(voiceControlsVisible("kokoro"), true);
  assert.equal(voiceControlsVisible("voiceforge"), false);
  assert.equal(voiceControlsVisible(""), false);
});

test("the voice is sent only for Kokoro", () => {
  assert.equal(voiceForRequest("kokoro", "af_heart"), "af_heart");
  assert.equal(voiceForRequest("kokoro", ""), null);
  assert.equal(voiceForRequest("voiceforge", "af_heart"), null);
});

test("voice metadata falls back gracefully", () => {
  assert.equal(
    describeVoice("kokoro", "af_heart", LABELS),
    "American female — warm and clear",
  );
  assert.equal(describeVoice("kokoro", "unknown", LABELS), "unknown");
  assert.equal(describeVoice("voiceforge", "af_heart", LABELS), "not applicable");
});
