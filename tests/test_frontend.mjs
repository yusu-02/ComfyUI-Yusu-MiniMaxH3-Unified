import assert from "node:assert/strict";
import fs from "node:fs";

const path = new URL("../web/minimax_h3_unified.js", import.meta.url);
const source = fs.readFileSync(path, "utf8")
    .replace(/^import .*;$/gm, "")
    .split("const STYLE_ID =")[0]
    + `
globalThis.h3Test = {
    migrateDurationInput, migrateWidgets, pruneModeInputs, normalizeOutputs,
    removeLegacyModelInputs, alignedFrameCountFromSeconds, roundHalfToEven,
    canonicalInputName, externalInputName, normalizeState, videoAudioAvailability,
    readJsonResponse, syncDurationControls, expandCollapsedAutogrowInputs,
    clipboardMediaFiles, firstOpenMediaSlot,
};`;
eval(source);

const legacyDuration = {
    inputs: [{ name: "length", label: "length", localized_name: "length", widget: { name: "length", label: "length" }, link: 31 }],
};
h3Test.migrateDurationInput(legacyDuration);
assert.equal(legacyDuration.inputs[0].name, "duration");
assert.equal(legacyDuration.inputs[0].link, 31);

const legacyWidgets = { widgets_values: ["omni_reference", "prompt", 1344, 768, 124, "match", "{}"] };
h3Test.migrateWidgets(legacyWidgets, 0);
assert.deepEqual(legacyWidgets.widgets_values, ["omni_reference", "match", "prompt", 1344, 768, 5.167, false, "{}"]);

assert.equal(h3Test.alignedFrameCountFromSeconds(0), 5);
assert.equal(h3Test.alignedFrameCountFromSeconds(5), 124);
assert.equal(h3Test.alignedFrameCountFromSeconds(20), 481);
assert.equal(h3Test.roundHalfToEven(124.5), 124);
assert.equal(h3Test.roundHalfToEven(125.5), 126);
assert.equal(h3Test.videoAudioAvailability({ has_audio: false }), "absent");

const durationNode = {
    widgets: [
        { name: "mode", value: "omni_reference" },
        { name: "auto_length_from_audio", value: true },
        { name: "duration", value: 5 },
    ],
    graph: { setDirtyCanvas() {} },
};
h3Test.syncDurationControls(durationNode);
assert.equal(durationNode.widgets[2].disabled, true);

function graphNode(inputs, outputs = []) {
    return {
        inputs,
        outputs,
        graph: { _links: new Map(), setDirtyCanvas() {} },
        removeInput(index) {
            const removed = this.inputs[index];
            if (removed?.link != null) this.graph._links.delete(removed.link);
            this.inputs.splice(index, 1);
            for (const link of this.graph._links.values()) {
                if (link.target_slot > index) link.target_slot -= 1;
            }
        },
        removeOutput(index) {
            const removed = this.outputs[index];
            for (const id of removed?.links || []) this.graph._links.delete(id);
            this.outputs.splice(index, 1);
            for (const link of this.graph._links.values()) {
                if (link.origin_slot > index) link.origin_slot -= 1;
            }
        },
        addOutput(name, type) {
            const output = { name, type, links: null };
            this.outputs.push(output);
            return output;
        },
    };
}

const inputNode = graphNode([
    { name: "model", link: 1 },
    { name: "clip", link: null },
    { name: "vae", link: null },
    { name: "audio_vae", link: null },
]);
inputNode.graph._links.set(1, { target_slot: 0 });
assert.equal(h3Test.removeLegacyModelInputs(inputNode), true);
assert.deepEqual(inputNode.inputs.map((item) => item.name), ["clip", "vae", "audio_vae"]);
assert.equal(inputNode.graph._links.has(1), false);

const legacyOutputs = graphNode([], [
    { name: "model", type: "MODEL", links: null },
    { name: "positive", type: "CONDITIONING", links: [11] },
    { name: "av_latent", type: "LATENT", links: [12] },
    { name: "audio", type: "AUDIO", links: [13] },
    { name: "media_info", type: "STRING", links: [14] },
]);
legacyOutputs.graph._links = new Map([
    [11, { origin_slot: 1 }],
    [12, { origin_slot: 2 }],
    [13, { origin_slot: 3 }],
    [14, { origin_slot: 4 }],
]);
assert.equal(h3Test.normalizeOutputs(legacyOutputs), true);
assert.deepEqual(legacyOutputs.outputs.map((item) => [item.name, item.type]), [
    ["positive", "CONDITIONING"],
    ["av_latent", "LATENT"],
    ["audio", "AUDIO"],
]);
assert.equal(legacyOutputs.graph._links.get(11).origin_slot, 0);
assert.equal(legacyOutputs.graph._links.get(12).origin_slot, 1);
assert.equal(legacyOutputs.graph._links.get(13).origin_slot, 2);
assert.equal(legacyOutputs.graph._links.has(14), false);

const modeNode = graphNode([
    { name: "clip", link: null },
    { name: "vae", link: null },
    { name: "audio_vae", link: 21 },
    { name: "first_frame", link: null },
    { name: "last_frame", link: null },
    { name: "ref_image_0", link: null },
]);
modeNode.widgets = [{ name: "mode", value: "text_to_video" }];
modeNode.graph._links.set(21, { target_slot: 2 });
h3Test.pruneModeInputs(modeNode);
assert.deepEqual(modeNode.inputs.map((item) => item.name), ["clip", "vae", "audio_vae"]);
assert.equal(modeNode.inputs[2].link, 21);

assert.deepEqual(h3Test.normalizeState("[1,2]"), {});
assert.equal(h3Test.externalInputName("ref_audio_1"), "ref_audio_0");

const pastedImage = { name: "clipboard.png" };
const pastedAudio = { name: "voice.wav" };
assert.deepEqual(h3Test.clipboardMediaFiles({
    items: [
        { kind: "string", getAsFile: () => null },
        { kind: "file", getAsFile: () => pastedImage },
        { kind: "file", getAsFile: () => pastedAudio },
        { kind: "file", getAsFile: () => ({ name: "notes.txt" }) },
    ],
}), [pastedImage, pastedAudio]);
const pasteNode = graphNode([
    { name: "mode.ref_images.ref_image_0", link: 30 },
]);
assert.equal(h3Test.firstOpenMediaSlot(pasteNode, {}, "image"), "ref_image_2");
assert.equal(h3Test.firstOpenMediaSlot(pasteNode, { ref_audio_1: { path: "voice.wav" } }, "audio"), "ref_audio_2");

const collapsedNode = graphNode([
    { name: "mode.ref_images", type: "*" },
    { name: "mode.ref_videos", type: "*" },
    { name: "mode.ref_video_audios", type: "*" },
    { name: "mode.ref_audios", type: "*" },
    { name: "mode.ref_images.ref_image_1", type: "IMAGE" },
]);
assert.equal(h3Test.expandCollapsedAutogrowInputs(collapsedNode), true);
assert.deepEqual(collapsedNode.inputs.map((item) => [item.name, item.type]), [
    ["mode.ref_images.ref_image_0", "IMAGE"],
    ["mode.ref_videos.ref_video_0", "IMAGE"],
    ["mode.ref_video_audios.ref_video_audio_0", "AUDIO"],
    ["mode.ref_audios.ref_audio_0", "AUDIO"],
    ["mode.ref_images.ref_image_1", "IMAGE"],
]);

assert.deepEqual(await h3Test.readJsonResponse({
    ok: true,
    status: 200,
    text: async () => '{"name":"a.wav"}',
}, "上传"), { name: "a.wav" });
await assert.rejects(() => h3Test.readJsonResponse({
    ok: false,
    status: 500,
    text: async () => "500 Internal Server Error",
}, "上传"), /服务端响应异常/);
await assert.rejects(() => h3Test.readJsonResponse({
    ok: true,
    status: 200,
    text: async () => "not-json",
}, "上传"), /服务端响应异常/);

const css = fs.readFileSync(new URL("../web/minimax_h3_unified.css", import.meta.url), "utf8");
assert.match(css, /grid-template-rows:22px 170px 30px/);
assert.match(css, /\.h3u-row-header/);
assert.match(css, /\.h3u-row-actions/);

console.log("frontend tests passed");
