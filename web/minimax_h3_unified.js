import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "MiniMaxH3Unified";
const IMAGE_SLOTS = Array.from({ length: 9 }, (_, i) => `ref_image_${i + 1}`);
const VIDEO_SLOTS = Array.from({ length: 3 }, (_, i) => `ref_video_${i + 1}`);
const PAIRED_AUDIO_SLOTS = Array.from({ length: 3 }, (_, i) => `ref_video_audio_${i + 1}`);
const AUDIO_SLOTS = Array.from({ length: 3 }, (_, i) => `ref_audio_${i + 1}`);
const LABELS = { first_frame: "首帧", last_frame: "尾帧" };
const FILE_EXTENSIONS = {
    image: new Set(["png", "jpg", "jpeg", "webp"]),
    video: new Set(["mp4", "webm", "mov", "mkv"]),
    audio: new Set(["wav", "mp3", "flac", "ogg", "m4a", "aac"]),
};
const H3_FPS = 24;
const MIN_OUTPUT_SECONDS = 0;
const DEFAULT_DURATION_SECONDS = 124 / H3_FPS;
const TRAINED_MAX_FRAMES = 362;
const MIN_LEGACY_FRAME_VALUE = 107;
const LEGACY_MAX_FRAME_VALUE = 362;
const WORKFLOW_SCHEMA_VERSION = 23;
const MEDIA_SUBDIR = "minimax_h3_unified";
const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

// Store only downsampled peaks, never decoded AudioBuffer objects. The old
// implementation fetched and decoded every audio file again whenever the panel
// rerendered (including while typing in prompt), which could spike browser RAM.
const WAVEFORM_CACHE = new Map();
const MAX_WAVEFORM_CACHE_ITEMS = 12;
const MAX_WAVEFORM_BYTES = 64 * 1024 * 1024;
const MAX_WAVEFORM_SECONDS = 300;

function fileKind(file) {
    const extension = String(file?.name || "").split(".").pop().toLowerCase();
    return Object.entries(FILE_EXTENSIONS).find(([, extensions]) => extensions.has(extension))?.[0] || null;
}

function clipboardMediaFiles(clipboardData) {
    const itemFiles = Array.from(clipboardData?.items || [])
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile?.())
        .filter(Boolean);
    return (itemFiles.length ? itemFiles : Array.from(clipboardData?.files || [])).filter(fileKind);
}

function firstOpenMediaSlot(node, state, kind) {
    const slots = kind === "image" ? IMAGE_SLOTS : kind === "video" ? VIDEO_SLOTS : AUDIO_SLOTS;
    return slots.find((slot) => !state[slot]?.path && !linked(node, slot)) || null;
}

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function roundHalfToEven(value) {
    if (!Number.isFinite(value)) return null;
    const lower = Math.floor(value);
    const fraction = value - lower;
    const tolerance = Number.EPSILON * Math.max(1, Math.abs(value)) * 4;
    if (Math.abs(fraction - 0.5) <= tolerance) {
        return lower % 2 === 0 ? lower : lower + 1;
    }
    return Math.round(value);
}

function alignedFrameCountFromSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return null;
    const rounded = roundHalfToEven(seconds * H3_FPS);
    if (!Number.isSafeInteger(rounded)) return null;
    let frameCount = Math.max(5, rounded);
    while (frameCount % 17 !== 5) frameCount += 1;
    return frameCount;
}

function videoAudioAvailability(item) {
    if (!item || typeof item !== "object") return "unknown";
    if (item.has_audio === false) return "absent";
    if (item.has_audio === true || Number(item.audio_duration) > 0) return "present";
    // Workflows saved before upload metadata included has_audio can still be
    // probed safely by the backend at execution time.
    return "unknown";
}

function setWidgetDisabled(item, disabled) {
    if (!item) return;
    item.disabled = Boolean(disabled);
    const targets = [item.inputEl, item.element, item.domElement, item.el]
        .filter((candidate, index, all) => candidate && all.indexOf(candidate) === index);
    for (const target of targets) {
        if ("disabled" in target) target.disabled = Boolean(disabled);
        target.setAttribute?.("aria-disabled", disabled ? "true" : "false");
        target.classList?.toggle("h3u-widget-disabled", Boolean(disabled));
    }
}

function syncDurationControls(node) {
    const mode = widget(node, "mode")?.value;
    const autoWidget = widget(node, "auto_length_from_audio");
    const durationWidget = widget(node, "duration");
    const autoAvailable = mode === "omni_reference";
    const autoEnabled = autoAvailable && Boolean(autoWidget?.value);

    const duration = Number(durationWidget?.value);
    if (durationWidget && (durationWidget.value === "" || !Number.isFinite(duration) || duration < MIN_OUTPUT_SECONDS)) {
        durationWidget.value = Number(DEFAULT_DURATION_SECONDS.toFixed(3));
    }

    setWidgetDisabled(autoWidget, !autoAvailable);
    setWidgetDisabled(durationWidget, autoEnabled);
    node.graph?.setDirtyCanvas?.(true, true);
    return { autoAvailable, autoEnabled };
}

function graphLink(graph, linkId) {
    return graph?._links?.get?.(linkId) ?? graph?.links?.[linkId] ?? null;
}

function transferInputLink(node, fromIndex, toIndex) {
    const from = node.inputs?.[fromIndex];
    const to = node.inputs?.[toIndex];
    if (!from || !to || from.link == null || to.link != null) return false;
    const linkId = from.link;
    const link = graphLink(node.graph, linkId);
    to.link = linkId;
    if (link) link.target_slot = toIndex;
    // removeInput() disconnects the link stored on the removed slot. Clear it
    // only after the destination and graph link have been updated.
    from.link = null;
    return true;
}

function canonicalInputName(name) {
    return String(name || "").split(".").pop();
}

function normalizeOutputs(node) {
    let changed = false;
    const desired = new Map([
        ["positive", "CONDITIONING"],
        ["av_latent", "LATENT"],
        ["audio", "AUDIO"],
    ]);

    // v18 and earlier exposed a MODEL passthrough followed by diagnostics.
    // Remove those sockets first so LiteGraph shifts the existing positive /
    // latent / audio links to their new slots instead of relabelling a MODEL
    // socket as CONDITIONING.
    for (let index = (node.outputs?.length || 0) - 1; index >= 0; index -= 1) {
        const output = node.outputs[index];
        const name = canonicalInputName(output?.name);
        if (name === "model" || String(output?.type || "").toUpperCase() === "MODEL" || !desired.has(name)) {
            node.removeOutput?.(index);
            changed = true;
        }
    }

    const ensure = (index, name, type) => {
        const current = node.outputs?.[index];
        if (!current) {
            const created = node.addOutput?.(name, type);
            if (created) {
                created.label = name;
                created.localized_name = name;
            }
            changed = true;
            return;
        }
        if (canonicalInputName(current.name) !== name || String(current.type || "") !== type) {
            // Unexpected old ordering: remove only this incompatible socket and
            // recreate it. Normal migrations retain links because removeOutput(0)
            // above shifts the official outputs into the expected positions.
            node.removeOutput?.(index);
            const created = node.addOutput?.(name, type);
            if (created) {
                created.label = name;
                created.localized_name = name;
            }
            changed = true;
            return;
        }
        current.name = name;
        current.label = name;
        current.localized_name = name;
    };

    ensure(0, "positive", "CONDITIONING");
    ensure(1, "av_latent", "LATENT");
    ensure(2, "audio", "AUDIO");
    while ((node.outputs?.length || 0) > 3) {
        node.removeOutput?.(node.outputs.length - 1);
        changed = true;
    }
    if (changed) node.graph?.setDirtyCanvas?.(true, true);
    return changed;
}

function externalInputName(name) {
    const match = String(name || "").match(/^(ref_video_audio|ref_image|ref_video|ref_audio)_(\d+)$/);
    return match ? `${match[1]}_${Number(match[2]) - 1}` : String(name || "");
}

function inputByName(node, name) {
    const target = externalInputName(name);
    return node.inputs?.find((input) => canonicalInputName(input.name) === target) || null;
}

function linked(node, name) {
    return inputByName(node, name)?.link != null;
}

function graphNodeById(graph, nodeId) {
    if (nodeId == null) return null;
    const direct = graph?.getNodeById?.(nodeId);
    if (direct) return direct;
    const store = graph?._nodes_by_id;
    const mapped = store?.get?.(nodeId) ?? store?.[nodeId];
    if (mapped) return mapped;
    return graph?._nodes?.find?.((candidate) => String(candidate?.id) === String(nodeId)) || null;
}

function stringWidgetValue(item) {
    return typeof item?.value === "string" ? item.value : null;
}

function promptSource(node) {
    const localWidget = widget(node, "prompt");
    const input = inputByName(node, "prompt");
    if (!input || input.link == null) {
        return { connected: false, readable: true, value: String(localWidget?.value || ""), widget: localWidget || null };
    }

    const link = graphLink(node.graph, input.link);
    if (!link) return { connected: true, readable: false, value: "", widget: null };
    const originId = link.origin_id ?? link.originId;
    const originSlot = Number(link.origin_slot ?? link.originSlot ?? 0);
    const origin = graphNodeById(node.graph, originId);
    if (!origin) return { connected: true, readable: false, value: "", widget: null };

    const output = origin.outputs?.[originSlot];
    const preferredNames = [
        output?.widget?.name,
        output?.name,
        output?.label,
        "text",
        "string",
        "value",
        "prompt",
    ].filter(Boolean).map(canonicalInputName);
    for (const name of preferredNames) {
        const sourceWidget = origin.widgets?.find((candidate) => canonicalInputName(candidate?.name) === name);
        const value = stringWidgetValue(sourceWidget);
        if (value != null) return { connected: true, readable: true, value, widget: sourceWidget };
    }

    const outputValue = origin.getOutputData?.(originSlot);
    if (typeof outputValue === "string") {
        return { connected: true, readable: true, value: outputValue, widget: null };
    }
    for (const key of ["text", "string", "value", "prompt"]) {
        if (typeof origin.properties?.[key] === "string") {
            return { connected: true, readable: true, value: origin.properties[key], widget: null };
        }
    }
    const stringWidgets = (origin.widgets || []).filter((candidate) => stringWidgetValue(candidate) != null);
    if (stringWidgets.length === 1) {
        return { connected: true, readable: true, value: stringWidgets[0].value, widget: stringWidgets[0] };
    }
    const serializedCandidates = [origin.widgets_values];
    for (const values of serializedCandidates) {
        const strings = Array.isArray(values) ? values.filter((value) => typeof value === "string") : [];
        if (strings.length === 1) return { connected: true, readable: true, value: strings[0], widget: null };
    }
    return { connected: true, readable: false, value: "", widget: null };
}

const WIDGET_CHANGE_HOOKS = new WeakMap();

function subscribeWidgetChanges(sourceWidget, listener) {
    if (!sourceWidget || typeof listener !== "function") return () => {};
    let hook = WIDGET_CHANGE_HOOKS.get(sourceWidget);
    if (!hook) {
        const original = sourceWidget.callback;
        hook = { original, listeners: new Set(), wrapper: null };
        hook.wrapper = function () {
            const result = hook.original?.apply(this, arguments);
            for (const callback of [...hook.listeners]) callback();
            return result;
        };
        sourceWidget.callback = hook.wrapper;
        WIDGET_CHANGE_HOOKS.set(sourceWidget, hook);
    }
    hook.listeners.add(listener);

    const domTargets = [sourceWidget.inputEl, sourceWidget.element, sourceWidget.domElement, sourceWidget.el]
        .filter((candidate, index, all) => candidate?.addEventListener && all.indexOf(candidate) === index);
    for (const target of domTargets) target.addEventListener("input", listener);

    return () => {
        for (const target of domTargets) target.removeEventListener("input", listener);
        const current = WIDGET_CHANGE_HOOKS.get(sourceWidget);
        current?.listeners.delete(listener);
        if (current && !current.listeners.size) {
            if (sourceWidget.callback === current.wrapper) sourceWidget.callback = current.original;
            WIDGET_CHANGE_HOOKS.delete(sourceWidget);
        }
    };
}

function subscribePromptSource(node, listener) {
    const source = promptSource(node);
    return source.connected && source.widget ? subscribeWidgetChanges(source.widget, listener) : () => {};
}

const INPUT_LABEL_RESTORE = new WeakMap();
const STATE_COMMIT_FRAMES = new WeakMap();

function restoreOriginalInputLabels(node) {
    const restorable = /^(?:clip|vae|audio_vae|first_frame|last_frame|ref_image_\d+|ref_video_\d+|ref_video_audio_\d+|ref_audio_\d+)$/;
    let changed = false;
    node.inputs?.forEach((input) => {
        const label = canonicalInputName(input.name);
        if (!restorable.test(label)) return;
        if (input.label !== label) {
            input.label = label;
            changed = true;
        }
        if (input.localized_name !== label) {
            input.localized_name = label;
            changed = true;
        }
        if (input.widget && input.widget.label !== label) {
            input.widget.label = label;
            changed = true;
        }
    });
    if (changed) node.graph?.setDirtyCanvas?.(true, true);
    return changed;
}

function scheduleOriginalInputLabels(node) {
    restoreOriginalInputLabels(node);
    const oldFrame = INPUT_LABEL_RESTORE.get(node);
    if (oldFrame) cancelAnimationFrame(oldFrame);
    let remaining = 2;
    const apply = () => {
        restoreOriginalInputLabels(node);
        remaining -= 1;
        if (remaining > 0) INPUT_LABEL_RESTORE.set(node, requestAnimationFrame(apply));
        else INPUT_LABEL_RESTORE.delete(node);
    };
    INPUT_LABEL_RESTORE.set(node, requestAnimationFrame(apply));
}

function removeLegacyModelInputs(node) {
    const remove = [];
    node.inputs?.forEach((input, index) => {
        const name = canonicalInputName(input?.name);
        if (name === "model" || name === "fl2va_model" || name === "ref2va_model") remove.push(index);
    });
    remove.sort((a,b)=>b-a).forEach((index)=> node.removeInput?.(index));
    if (remove.length) node.graph?.setDirtyCanvas?.(true, true);
    return Boolean(remove.length);
}

function migrateDurationInput(info) {
    if (!Array.isArray(info?.inputs)) return false;
    let changed = false;
    for (const input of info.inputs) {
        if (canonicalInputName(input?.name) !== "length") continue;
        const rawName = String(input.name);
        const prefix = rawName.includes(".")
            ? rawName.slice(0, rawName.lastIndexOf(".") + 1)
            : "";
        input.name = `${prefix}duration`;
        if (input.localized_name === "length") input.localized_name = "duration";
        if (input.label === "length") input.label = "duration";
        if (input.widget?.name === "length") input.widget.name = "duration";
        if (input.widget?.label === "length") input.widget.label = "duration";
        changed = true;
    }
    return changed;
}

function migrateWidgets(info, savedVersion = 0) {
    let values = info?.widgets_values;
    if (!Array.isArray(values)) return;
    values = [...values];

    const mode = String(values[0] || "text_to_video");
    const version = Number(savedVersion || 0);

    // v16 temporarily made ref_image_size a persistent widget in every mode.
    // v17 restores mode-specific widgets, so non-reference workflows must drop
    // that extra value while omni_reference keeps it at index 1.
    if (version === 16) {
        if (mode !== "omni_reference" && ["match", "max"].includes(values[1])) {
            values.splice(1, 1);
        }
    } else if (["match", "max"].includes(values[5])) {
        // Older releases stored ref_image_size after duration.
        values = mode === "omni_reference"
            ? [values[0], values[5], values[1], values[2], values[3], values[4], values[6]]
            : [values[0], values[1], values[2], values[3], values[4], values[6]];
    }

    if (typeof values[values.length - 1] === "string" && typeof values[values.length - 2] !== "boolean") {
        values.splice(values.length - 1, 0, false);
    }

    // v11 and earlier stored frame counts. Integer seconds are valid from v15.
    const durationIndex = values.length - 3;
    const legacyValue = Number(values[durationIndex]);
    if (
        version < 15
        && Number.isInteger(legacyValue)
        && legacyValue >= MIN_LEGACY_FRAME_VALUE
        && legacyValue <= LEGACY_MAX_FRAME_VALUE
    ) {
        values[durationIndex] = Number((legacyValue / H3_FPS).toFixed(3));
    }
    info.widgets_values = values;
}

function expandCollapsedAutogrowInputs(node) {
    const groups = {
        ref_images: ["ref_image_0", "IMAGE"],
        ref_videos: ["ref_video_0", "IMAGE"],
        ref_video_audios: ["ref_video_audio_0", "AUDIO"],
        ref_audios: ["ref_audio_0", "AUDIO"],
    };
    let changed = false;
    node.inputs?.forEach((input) => {
        const target = groups[canonicalInputName(input?.name)];
        if (!target) return;
        const [child, type] = target;
        input.name = `${input.name}.${child}`;
        input.type = type;
        input.label = child;
        input.localized_name = child;
        changed = true;
    });
    if (changed) node.graph?.setDirtyCanvas?.(true, true);
    return changed;
}

function pruneModeInputs(node) {
    const mode = widget(node, "mode")?.value;
    const modeInput = /^(?:first_frame|last_frame|ref_image_\d+|ref_video_\d+|ref_video_audio_\d+|ref_audio_\d+)$/;
    const allowed = (name) => mode === "first_last_frame"
        ? name === "first_frame" || name === "last_frame"
        : mode === "omni_reference" && /^(?:audio_vae|ref_image_\d+|ref_video_\d+|ref_video_audio_\d+|ref_audio_\d+)$/.test(name);
    const kept = new Map();
    const remove = [];
    node.inputs?.forEach((input, index) => {
        const name = canonicalInputName(input.name);
        if (!modeInput.test(name)) return;
        if (!allowed(name) || kept.has(name)) {
            const target = kept.get(name);
            if (target != null) transferInputLink(node, index, target);
            remove.push(index);
        } else {
            kept.set(name, index);
        }
    });
    remove.reverse().forEach((index) => node.removeInput(index));
    restoreOriginalInputLabels(node);
}

function viewUrl(path) {
    const parts = path.split("/");
    const filename = parts.pop();
    return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(parts.join("/"))}&type=input`);
}

function element(tag, properties = {}, children = []) {
    const item = document.createElement(tag);
    Object.assign(item, properties);
    item.append(...children);
    return item;
}

function button(text, action) {
    return element("button", { textContent: text, onclick: action, className: "h3u-button" });
}

function hideWidget(widget) {
    // Modern ComfyUI lays DOM/Vue widgets out through `hidden`; old LiteGraph
    // builds only honor computeSize/draw. Apply both paths while leaving the
    // widget serializable so media_state is still saved in the workflow.
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    for (const candidate of [widget.element, widget.inputEl, widget.domElement, widget.el]) {
        if (!(candidate instanceof HTMLElement)) continue;
        candidate.style.setProperty("display", "none", "important");
        candidate.style.setProperty("height", "0", "important");
        candidate.style.setProperty("min-height", "0", "important");
        const host = candidate.closest(".dom-widget");
        host?.style.setProperty("display", "none", "important");
        host?.style.setProperty("height", "0", "important");
        host?.style.setProperty("min-height", "0", "important");
    }
}

function setState(node, stateWidget, state) {
    const serialized = JSON.stringify(state);
    if (stateWidget.value === serialized) return;
    stateWidget.value = serialized;
    const pending = STATE_COMMIT_FRAMES.get(node);
    if (pending) cancelAnimationFrame(pending);
    STATE_COMMIT_FRAMES.set(node, requestAnimationFrame(() => {
        STATE_COMMIT_FRAMES.delete(node);
        stateWidget.callback?.(stateWidget.value);
        node.graph?.setDirtyCanvas?.(true, true);
    }));
}

async function upload(accept) {
    const input = element("input", { type: "file", accept });
    const file = await new Promise((resolve) => {
        let settled = false;
        const finish = (value) => {
            if (settled) return;
            settled = true;
            resolve(value);
        };
        input.onchange = () => finish(input.files?.[0]);
        input.oncancel = () => finish(null);
        input.click();
    });
    return file ? uploadFile(file) : null;
}

async function readJsonResponse(response, label) {
    const text = await response.text();
    let result;
    try { result = JSON.parse(text); } catch { result = null; }
    if (!response.ok) {
        if (result?.error) throw new Error(result.error);
        throw new Error(`${label}失败 (${response.status})，服务端响应异常，请重启 ComfyUI 后重试`);
    }
    if (!result || Array.isArray(result) || typeof result !== "object") {
        throw new Error(`${label}服务端响应异常 (${response.status})，请重启 ComfyUI 后重试`);
    }
    return result;
}

async function uploadFile(file) {
    const kind = fileKind(file);
    if (!kind) throw new Error("不支持的媒体文件格式");
    if (file.size > MAX_UPLOAD_BYTES) throw new Error("文件超过 512 MiB 上限");
    const data = new FormData();
    data.append("image", file);
    data.append("type", "input");
    data.append("subfolder", MEDIA_SUBDIR);
    const uploadResponse = await api.fetchApi("/upload/image", { method: "POST", body: data });
    const uploaded = await readJsonResponse(uploadResponse, "上传");
    if (typeof uploaded.name !== "string" || !uploaded.name) {
        throw new Error("上传服务未返回文件名，请重启 ComfyUI 后重试");
    }
    const path = [uploaded.subfolder, uploaded.name].filter(Boolean).join("/");
    const inspectResponse = await api.fetchApi("/minimax_h3_unified/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, kind, mime: file.type }),
    });
    if (!inspectResponse.ok) {
        if (inspectResponse.status === 404 || inspectResponse.status === 405) {
            throw new Error("插件后端仍是旧版本，请完全重启 ComfyUI 后再刷新浏览器");
        }
    }
    const result = await readJsonResponse(inspectResponse, "媒体检查");
    return { ...result, name: file.name };
}

function dropZone(kind, choose) {
    const names = { image: "图片", video: "视频", audio: "音频" };
    const zone = element("div", {
        className: `h3u-drop h3u-drop-${kind}`,
        tabIndex: 0,
        role: "button",
        ariaLabel: `拖拽或选择${names[kind]}`,
    }, [
        element("span", { className: "h3u-drop-icon", textContent: kind === "image" ? "▧" : kind === "video" ? "▶" : "♫" }),
        element("span", { textContent: kind === "image" ? "拖拽图片到这里" : `拖拽${names[kind]}到轨道` }),
        element("small", { textContent: "或点击选择" }),
    ]);
    zone.onclick = () => choose();
    zone.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            choose();
        }
    };
    zone.onpointerdown = (event) => event.stopPropagation();
    zone.ondragenter = zone.ondragover = (event) => {
        event.preventDefault();
        event.stopPropagation();
        zone.classList.add("h3u-drop-active");
    };
    zone.ondragleave = () => zone.classList.remove("h3u-drop-active");
    zone.ondrop = (event) => {
        event.preventDefault();
        event.stopPropagation();
        zone.classList.remove("h3u-drop-active");
        const file = event.dataTransfer?.files?.[0];
        if (file) choose(file);
    };
    return zone;
}

function trimControls(node, stateWidget, state, slot, item, media, kind, url) {
    const rawDuration = Number(item.duration || 0);
    const duration = Number.isFinite(rawDuration) ? Math.max(0, rawDuration) : 0;
    const rawStart = Number(item.trim_start || 0);
    const rawEnd = Number(item.trim_end || duration);
    const start = Number.isFinite(rawStart) ? Math.min(duration, Math.max(0, rawStart)) : 0;
    const end = Number.isFinite(rawEnd) ? Math.min(duration, Math.max(start, rawEnd)) : duration;
    const controls = element("div", { className: `h3u-track h3u-track-${kind}` });
    controls.onpointerdown = controls.onmousedown = controls.ontouchstart = (event) => event.stopPropagation();
    const timeline = element("div", { className: "h3u-timeline" });
    const selection = element("div", { className: "h3u-track-selection" });
    const startRange = element("input", { type: "range", min: 0, max: duration, step: 0.001, value: start, className: "h3u-track-range h3u-track-start" });
    const endRange = element("input", { type: "range", min: 0, max: duration, step: 0.001, value: end, className: "h3u-track-range h3u-track-end" });
    const startNumber = element("input", { type: "number", min: 0, max: duration, step: 0.001, value: start, className: "h3u-number" });
    const endNumber = element("input", { type: "number", min: 0, max: duration, step: 0.001, value: end, className: "h3u-number" });
    const durationText = element("span", { className: "h3u-duration" });
    if (kind === "audio") {
        const canvas = element("canvas", { width: 1000, height: 70, className: "h3u-track-wave" });
        canvas.onpointerdown = (event) => event.stopPropagation();
        timeline.append(canvas);
        setupWaveformPreview(canvas, url, item);
    } else {
        timeline.append(element("span", { className: "h3u-video-track-label", textContent: "视频轨道 · 24 FPS" }));
    }
    timeline.append(selection, startRange, endRange);

    const update = (source, isStart) => {
        let value = Math.min(duration, Math.max(0, Number(source.value) || 0));
        if (isStart) value = Math.min(value, Number(endRange.value) - 0.001);
        else value = Math.max(value, Number(startRange.value) + 0.001);
        (isStart ? startRange : endRange).value = value;
        (isStart ? startNumber : endNumber).value = value.toFixed(3);
        item[isStart ? "trim_start" : "trim_end"] = value;
        syncTrack();
        setState(node, stateWidget, state);
    };
    const syncTrack = () => {
        const left = duration ? Number(startRange.value) / duration * 100 : 0;
        const right = duration ? Number(endRange.value) / duration * 100 : 100;
        selection.style.left = `${left}%`;
        selection.style.width = `${Math.max(0, right - left)}%`;
        durationText.textContent = `选区 ${(Number(endRange.value) - Number(startRange.value)).toFixed(3)}s`;
    };
    startRange.oninput = startNumber.onchange = () => update(startRange === document.activeElement ? startRange : startNumber, true);
    endRange.oninput = endNumber.onchange = () => update(endRange === document.activeElement ? endRange : endNumber, false);
    timeline.onpointerdown = (event) => {
        if (event.button != null && event.button !== 0) return;
        event.stopPropagation();
        const bounds = timeline.getBoundingClientRect();
        const valueAt = (pointer) => duration * Math.min(1, Math.max(0, (pointer.clientX - bounds.left) / Math.max(1, bounds.width)));
        const initial = valueAt(event);
        const isStart = Math.abs(initial - Number(startRange.value)) <= Math.abs(initial - Number(endRange.value));
        const move = (pointer) => {
            pointer.stopPropagation();
            const range = isStart ? startRange : endRange;
            range.value = valueAt(pointer);
            update(range, isStart);
        };
        const stop = (pointer) => {
            pointer.stopPropagation();
            timeline.removeEventListener("pointermove", move);
            timeline.removeEventListener("pointerup", stop);
            timeline.removeEventListener("pointercancel", stop);
        };
        timeline.setPointerCapture?.(event.pointerId);
        timeline.addEventListener("pointermove", move);
        timeline.addEventListener("pointerup", stop);
        timeline.addEventListener("pointercancel", stop);
        move(event);
    };
    controls.append(
        element("div", { className: "h3u-track-head" }, [
            element("strong", { textContent: kind === "video" ? "视频轨道" : "音频轨道" }),
            durationText,
        ]),
        timeline,
        element("div", { className: "h3u-track-times" }, [
            element("label", {}, [document.createTextNode("入点 "), startNumber]),
            element("span", { textContent: `总长 ${duration.toFixed(3)}s` }),
            element("label", {}, [document.createTextNode("出点 "), endNumber]),
        ]),
    );
    syncTrack();

    if (media instanceof HTMLMediaElement) {
        const play = button("播放选区", () => {
            media.currentTime = Number(startRange.value);
            media.play();
        });
        media.ontimeupdate = () => {
            if (media.currentTime >= Number(endRange.value)) media.pause();
        };
        controls.append(element("div", { className: "h3u-track-actions" }, [play, button("暂停", () => media.pause())]));
    }
    controls.querySelector(".h3u-track-actions")?.append(button("重置", () => {
        startRange.value = startNumber.value = 0;
        endRange.value = endNumber.value = duration;
        item.trim_start = 0;
        item.trim_end = duration;
        syncTrack();
        setState(node, stateWidget, state);
    }));
    return controls;
}

function paintWaveformMessage(canvas, text) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#8794a8";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

function paintWaveform(canvas, peaks) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#72a7ff";
    ctx.beginPath();
    for (let x = 0; x < peaks.length; x++) {
        const peak = peaks[x];
        ctx.moveTo(x, canvas.height / 2 - peak * canvas.height / 2);
        ctx.lineTo(x, canvas.height / 2 + peak * canvas.height / 2);
    }
    ctx.stroke();
}

async function waveformPeaks(url, width) {
    const key = `${url}|${width}`;
    if (WAVEFORM_CACHE.has(key)) return WAVEFORM_CACHE.get(key);
    const promise = (async () => {
        const response = await fetch(url, { cache: "force-cache" });
        if (!response.ok) throw new Error(`波形文件读取失败 (${response.status})`);
        const bytes = await response.arrayBuffer();
        const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
        if (!AudioContextClass) throw new Error("浏览器不支持 AudioContext");
        const context = new AudioContextClass();
        try {
            const audio = await context.decodeAudioData(bytes);
            const data = audio.getChannelData(0);
            const peaks = new Float32Array(width);
            const step = Math.max(1, Math.ceil(data.length / width));
            for (let x = 0; x < width; x++) {
                let peak = 0;
                const end = Math.min(data.length, (x + 1) * step);
                for (let i = x * step; i < end; i++) peak = Math.max(peak, Math.abs(data[i]));
                peaks[x] = peak;
            }
            return peaks;
        } finally {
            await context.close().catch(() => {});
        }
    })();
    WAVEFORM_CACHE.set(key, promise);
    if (WAVEFORM_CACHE.size > MAX_WAVEFORM_CACHE_ITEMS) {
        WAVEFORM_CACHE.delete(WAVEFORM_CACHE.keys().next().value);
    }
    try {
        return await promise;
    } catch (error) {
        WAVEFORM_CACHE.delete(key);
        throw error;
    }
}

function setupWaveformPreview(canvas, url, item) {
    const fileBytes = Number(item?.size || 0);
    const duration = Number(item?.duration || 0);
    if ((fileBytes > 0 && fileBytes > MAX_WAVEFORM_BYTES) || (duration > 0 && duration > MAX_WAVEFORM_SECONDS)) {
        paintWaveformMessage(canvas, "文件较长，已禁用整段波形解码");
        canvas.title = "可使用上方播放器定位；为避免占用大量内存，不在浏览器中解码整段波形";
        return;
    }
    let loading = false;
    paintWaveformMessage(canvas, "点击加载波形（避免自动解码占用内存）");
    canvas.title = "点击后才会读取并解码音频波形";
    canvas.style.cursor = "pointer";
    canvas.onclick = async (event) => {
        event.stopPropagation();
        if (loading) return;
        loading = true;
        paintWaveformMessage(canvas, "正在加载波形…");
        try {
            const peaks = await waveformPeaks(url, canvas.width);
            if (canvas.isConnected) paintWaveform(canvas, peaks);
        } catch {
            if (canvas.isConnected) paintWaveformMessage(canvas, "无法解析波形");
            canvas.title = "浏览器无法解析该音频波形";
        } finally {
            loading = false;
        }
    };
}

function mediaRow(node, stateWidget, state, slot, kind, rerender) {
    const item = state[slot];
    const row = element("section", { className: "h3u-row" });
    const ordinal = Number(slot.match(/\d+$/)?.[0] || 0);
    const isPairedAudio = slot.startsWith("ref_video_audio_");
    const label = LABELS[slot] || (kind === "image"
        ? `参考图 ${ordinal}`
        : kind === "video"
            ? `参考视频 ${ordinal}`
            : isPairedAudio
                ? `视频 ${ordinal} 配对音频`
                : `独立音频 ${ordinal}`);
    const header = element("div", { className: "h3u-row-header" }, [
        element("strong", { textContent: label }),
    ]);
    if (linked(node, slot)) header.append(element("span", {
        textContent: `官方外部端口 ${externalInputName(slot)} 已覆盖本槽`,
        className: "h3u-linked",
    }));
    row.append(header);
    const accept = kind === "image" ? ".png,.jpg,.jpeg,.webp" : kind === "video" ? ".mp4,.webm,.mov,.mkv" : ".wav,.mp3,.flac,.ogg,.m4a,.aac";
    const choose = async (file = null) => {
        try {
            if (file) {
                const detected = fileKind(file);
                if (detected && detected !== kind) {
                    throw new Error(`该槽位需要${kind === "image" ? "图片" : kind === "video" ? "视频" : "音频"}，拖入的文件扩展名属于${detected === "image" ? "图片" : detected === "video" ? "视频" : "音频"}`);
                }
            }
            const uploaded = file ? await uploadFile(file) : await upload(accept);
            if (!uploaded) return;
            if (uploaded.kind !== kind) {
                throw new Error(`该槽位需要${kind === "image" ? "图片" : kind === "video" ? "视频" : "音频"}，实际上传的是${uploaded.kind || "未知媒体"}`);
            }
            state[slot] = { ...uploaded, trim_start: 0, trim_end: uploaded.duration || 0, use_audio: false };
            setState(node, stateWidget, state);
            rerender();
        } catch (error) {
            alert(error.message);
        }
    };
    if (!item) {
        row.append(dropZone(kind, choose));
        return row;
    }

    header.append(element("span", { textContent: `${item.name} · ${Number(item.duration || 0).toFixed(3)}s`, className: "h3u-name" }));
    const url = viewUrl(item.path);
    let media;
    if (kind === "image") media = element("img", { src: url, loading: "lazy", decoding: "async", className: "h3u-preview" });
    if (kind === "video") media = element("video", { src: url, controls: true, preload: "metadata", className: "h3u-preview", onclick: (event) => event.stopPropagation() });
    if (kind === "audio") {
        media = element("audio", { src: url, controls: true, preload: "metadata", className: "h3u-audio", onclick: (event) => event.stopPropagation() });
    }
    const preview = dropZone(kind, choose);
    preview.classList.add("h3u-drop-filled");
    preview.replaceChildren(media, element("span", { className: "h3u-drop-replace", textContent: "拖拽可替换" }));
    row.append(preview);
    if (kind !== "image") row.append(trimControls(node, stateWidget, state, slot, item, media, kind, url));
    if (kind === "video") {
        const audioAvailability = videoAudioAvailability(item);
        if (audioAvailability === "absent") {
            item.use_audio = false;
            row.append(element("span", {
                className: "h3u-note",
                textContent: "该视频不含音轨，不能启用“使用视频原声”。",
            }));
        } else {
            const useAudio = element("input", { type: "checkbox", checked: Boolean(item.use_audio) });
            useAudio.onchange = () => {
                item.use_audio = useAudio.checked;
                setState(node, stateWidget, state);
                rerender();
            };
            const suffix = audioAvailability === "unknown" ? "（运行时确认音轨）" : "";
            row.append(element("label", { className: "h3u-check" }, [
                useAudio,
                document.createTextNode(` 使用视频原声${suffix}`),
            ]));
        }
    }
    row.append(element("div", { className: "h3u-row-actions" }, [
        button("替换", () => choose()),
        button("删除", () => {
            delete state[slot];
            setState(node, stateWidget, state);
            rerender();
        }),
    ]));
    return row;
}

function mapping(node, state) {
    const tags = [];
    const active = (slot) => linked(node, slot) || Boolean(state[slot]?.path);
    const source = (slot) => linked(node, slot) ? `外部 ${externalInputName(slot)}` : state[slot]?.name || "节点内文件";
    IMAGE_SLOTS.filter(active).forEach((slot, index) => tags.push(`<Picture ${index + 1}> ← ${slot} (${source(slot)})`));
    let audioIndex = 1;
    let videoIndex = 1;
    VIDEO_SLOTS.filter(active).forEach((slot) => {
        const audioSlot = slot.replace("ref_video_", "ref_video_audio_");
        if (active(audioSlot)) tags.push(`<Audio ${audioIndex++}> ← ${audioSlot} (${source(audioSlot)})`);
        else if (
            !linked(node, slot)
            && state[slot]?.use_audio
            && videoAudioAvailability(state[slot]) !== "absent"
        ) tags.push(`<Audio ${audioIndex++}> ← ${slot} 原声`);
        tags.push(`<Video ${videoIndex++}> ← ${slot} (${source(slot)})`);
    });
    AUDIO_SLOTS.filter(active).forEach((slot) => tags.push(`<Audio ${audioIndex++}> ← ${slot} (${source(slot)})`));
    return tags.join("\n") || "暂无节点内参考素材";
}

function slotCount(state, key, slots, node = null, relatedSlots = []) {
    const used = slots.reduce((highest, slot, index) => {
        const related = relatedSlots[index];
        return state[slot]?.path || (node && linked(node, slot)) || (related && (state[related]?.path || (node && linked(node, related))))
            ? index + 1
            : highest;
    }, 0);
    const saved = Number(state._ui?.[key]);
    if (Number.isInteger(saved)) {
        // Never hide a still-active saved reference or connected socket behind
        // a stale UI count. This is especially important for paired soundtracks.
        return Math.min(slots.length, Math.max(used, Math.max(0, saved)));
    }
    return Math.max(1, used);
}

function countSelector(node, stateWidget, state, label, key, slots, rerender, relatedSlots = []) {
    const select = element("select", { className: "h3u-count" });
    for (let count = 0; count <= slots.length; count++) {
        select.append(element("option", { value: count, textContent: count, selected: count === slotCount(state, key, slots, node, relatedSlots) }));
    }
    select.onchange = () => {
        const count = Number(select.value);
        state._ui = { ...state._ui, [key]: count };
        slots.slice(count).forEach((slot, offset) => {
            delete state[slot];
            const related = relatedSlots[count + offset];
            if (related) delete state[related];
        });
        setState(node, stateWidget, state);
        rerender();
    };
    return element("label", { className: "h3u-count-label" }, [document.createTextNode(label), select]);
}

function normalizeState(value) {
    let parsed;
    try { parsed = typeof value === "string" ? JSON.parse(value || "{}") : value; } catch { parsed = {}; }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") parsed = {};
    const state = {};
    for (const [key, item] of Object.entries(parsed)) {
        if (key === "_ui" && item && !Array.isArray(item) && typeof item === "object") {
            state._ui = item;
        } else if (item && !Array.isArray(item) && typeof item === "object" && typeof item.path === "string") {
            state[key] = item;
        }
    }
    return state;
}

function audioReferenceStatus(node, audioTags, promptValue = promptSource(node)) {
    const source = promptValue && typeof promptValue === "object" && "readable" in promptValue
        ? promptValue
        : { connected: false, readable: true, value: String(promptValue || "") };
    const audioVaeConnected = linked(node, "audio_vae");

    if (!source.readable) {
        if (audioVaeConnected) {
            return {
                ok: null,
                message: `audio_vae 已连接，prompt 也已通过外部端口连接。上游文本可能是动态生成，前端无法可靠读取；运行时会按实际 prompt 校验 ${audioTags.join("、")}，此处不再误报缺失。`,
            };
        }
        return {
            ok: false,
            message: `prompt 已通过外部端口连接，但 audio_vae 未连接，参考音频无法编码。`,
        };
    }

    const promptText = String(source.value || "").toLowerCase();
    const missing = audioTags.filter((tag) => !promptText.includes(tag.toLowerCase()));
    if (audioVaeConnected && !missing.length) {
        return {
            ok: true,
            message: `audio_vae 已连接，prompt 已引用：${audioTags.join("、")}。已满足参考音频编码条件。`,
        };
    }
    if (audioVaeConnected) {
        return {
            ok: false,
            message: `audio_vae 已连接，参考音频也已加入槽位；但 prompt 还缺少 ${missing.join("、")}。请在 prompt 中显式使用这些标签。`,
        };
    }
    if (!missing.length) {
        return {
            ok: false,
            message: `prompt 已引用：${audioTags.join("、")}；但 audio_vae 未连接，参考音频无法编码。`,
        };
    }
    return {
        ok: false,
        message: `参考音频已加入槽位；audio_vae 未连接，并且 prompt 还缺少 ${missing.join("、")}。`,
    };
}

function buildPanel(node, stateWidget) {
    let state = {};
    let resizeFrame = 0;
    let renderFrame = 0;
    let audioStatusElement = null;
    let audioStatusTags = [];
    let promptSourceUnsubscribe = () => {};
    const restoreState = () => {
        state = normalizeState(stateWidget.value);
        const normalized = JSON.stringify(state);
        if (stateWidget.value !== normalized) stateWidget.value = normalized;
    };
    restoreState();
    const root = element("div", {
        className: "h3u-panel-host",
        tabIndex: 0,
        title: "点击面板后可按 Ctrl+V 粘贴图片、视频或音频",
    });
    const content = element("div", { className: "h3u-panel" });
    root.append(content);
    const panelHeight = () => Math.max(24, Math.ceil(content.scrollHeight || content.getBoundingClientRect().height || 0) + 12);
    const syncPanelWidth = () => {
        const nodeWidth = Math.max(Number(node.size?.[0]) || 0, 560);
        root.style.width = `${Math.max(0, nodeWidth - 20)}px`;
    };
    syncPanelWidth();
    const domWidget = node.addDOMWidget("h3_media_panel", "h3-media", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: panelHeight,
        getMaxHeight: panelHeight,
        getHeight: panelHeight,
        afterResize: syncPanelWidth,
    });
    domWidget.computeSize = (width) => [Math.max(Number(width) || 0, 560), panelHeight()];
    const fitNode = () => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(() => {
            hideWidget(stateWidget);
            const width = Math.max(Number(node.size?.[0]) || 0, 560);
            syncPanelWidth();
            // Old workflows can persist a very large node height. Cap the DOM
            // widget and compute from a collapsed height so LiteGraph does not
            // feed that stale height back into the new layout.
            if (node.size) node.size[1] = 1;
            const natural = node.computeSize([width, 1]);
            node.setSize([width, Math.ceil(natural[1])]);
            node.graph?.setDirtyCanvas(true, true);
        });
    };

    const updateAudioStatus = () => {
        if (!audioStatusElement || !audioStatusTags.length) return;
        const status = audioReferenceStatus(node, audioStatusTags, promptSource(node));
        audioStatusElement.className = status.ok === true ? "h3u-audio-ok" : status.ok === false ? "h3u-warning" : "h3u-info";
        audioStatusElement.textContent = status.message;
    };

    const render = () => {
        content.replaceChildren();
        audioStatusElement = null;
        audioStatusTags = [];
        const mode = widget(node, "mode")?.value;
        const durationState = syncDurationControls(node);
        const durationWidget = widget(node, "duration");
        if (!durationState.autoEnabled && durationWidget) {
            const requestedSeconds = Number(durationWidget.value);
            const effectiveFrames = alignedFrameCountFromSeconds(requestedSeconds);
            if (
                Number.isFinite(requestedSeconds)
                && requestedSeconds >= MIN_OUTPUT_SECONDS
                && effectiveFrames != null
            ) {
                content.append(element("div", {
                    className: "h3u-info h3u-duration-preview",
                    textContent: `手动目标 ${requestedSeconds.toFixed(3)} 秒 → ${effectiveFrames} 帧 → 实际 ${(
                        effectiveFrames / H3_FPS
                    ).toFixed(3)} 秒${effectiveFrames > TRAINED_MAX_FRAMES ? "（未限制，已超过常见训练范围）" : ""}。`,
                }));
            }
        }
        if (mode === "text_to_video") {
            content.append(element("div", { textContent: "文生视频模式：媒体输入会被忽略。", className: "h3u-note" }));
        } else if (mode === "first_last_frame") {
            content.append(element("h3", { textContent: "首尾帧（外部端口逐槽覆盖节点内素材）" }));
            content.append(element("div", { className: "h3u-media-grid h3u-frame-grid" }, [
                mediaRow(node, stateWidget, state, "first_frame", "image", scheduleRender),
                mediaRow(node, stateWidget, state, "last_frame", "image", scheduleRender),
            ]));
        } else {
            const imageCount = IMAGE_SLOTS.filter((slot) => linked(node, slot) || state[slot]?.path).length;
            const videoCount = VIDEO_SLOTS.filter((slot) => linked(node, slot) || state[slot]?.path).length;
            const pairedAudioCount = PAIRED_AUDIO_SLOTS.filter((slot) => linked(node, slot) || state[slot]?.path).length;
            const audioCount = AUDIO_SLOTS.filter((slot) => linked(node, slot) || state[slot]?.path).length;
            content.append(element("h3", { textContent: `全模态参考 · 图片 ${imageCount}/9 · 视频 ${videoCount}/3 · 配对音频 ${pairedAudioCount}/3 · 独立音频 ${audioCount}/3` }));
            content.append(element("div", {
                className: "h3u-info",
                textContent: "点击本面板后按 Ctrl+V，可将剪贴板中的图片、视频或音频自动加入第一个空槽位。",
            }));
            const imageSlotCount = slotCount(state, "image_count", IMAGE_SLOTS, node);
            const videoSlotCount = slotCount(state, "video_count", VIDEO_SLOTS, node, PAIRED_AUDIO_SLOTS);
            const audioSlotCount = slotCount(state, "audio_count", AUDIO_SLOTS, node);
            content.append(element("div", { className: "h3u-counts" }, [
                element("strong", { textContent: "节点内槽位" }),
                countSelector(node, stateWidget, state, "参考图", "image_count", IMAGE_SLOTS, scheduleRender),
                countSelector(node, stateWidget, state, "参考视频", "video_count", VIDEO_SLOTS, scheduleRender, PAIRED_AUDIO_SLOTS),
                countSelector(node, stateWidget, state, "独立音频", "audio_count", AUDIO_SLOTS, scheduleRender),
            ]));
            content.append(element("div", {
                className: "h3u-info h3u-port-note",
                textContent: "官方 ref_video_audio_0～ref_video_audio_2 是同编号参考视频的配对音轨：ref_video_audio_1 对应 ref_video_1。节点内“视频 1～3 配对音频”与这些接口逐一对应；同槽同时存在时，外部接口优先。ref_audio_0～ref_audio_2 则是独立参考音频。",
            }));
            const mapText = mapping(node, state);
            content.append(element("pre", { textContent: mapText, className: "h3u-map" }));
            audioStatusTags = [...mapText.matchAll(/<Audio \d+>/g)].map((match) => match[0]);
            if (audioStatusTags.length) {
                audioStatusElement = element("div");
                content.append(audioStatusElement);
                updateAudioStatus();
            }
            if (Boolean(widget(node, "auto_length_from_audio")?.value)) {
                const hasUsableAudioReference = audioStatusTags.length > 0;
                content.append(element("div", {
                    className: "h3u-info",
                    textContent: hasUsableAudioReference
                        ? "已启用有音频时自动长度：手动时长已锁定。运行时读取裁剪后的实际音频时长，以最长一段为基准，按 24 FPS 对齐到 H3 的 17k+5 帧网格。"
                        : "已启用有音频时自动长度：手动时长已锁定；当前没有有效参考音频，执行时会回退到锁定前保存的手动秒数。",
                }));
            }
            content.append(element("div", { className: "h3u-media-grid h3u-image-grid" }, IMAGE_SLOTS.slice(0, imageSlotCount).map((slot) => mediaRow(node, stateWidget, state, slot, "image", scheduleRender))));
            const videoPairs = VIDEO_SLOTS.slice(0, videoSlotCount).map((slot, index) => {
                const audioSlot = PAIRED_AUDIO_SLOTS[index];
                const pair = element("div", { className: "h3u-video-pair" }, [
                    mediaRow(node, stateWidget, state, slot, "video", scheduleRender),
                    mediaRow(node, stateWidget, state, audioSlot, "audio", scheduleRender),
                ]);
                if (!(linked(node, slot) || state[slot]?.path) && (linked(node, audioSlot) || state[audioSlot]?.path)) {
                    pair.prepend(element("div", {
                        className: "h3u-warning",
                        textContent: `${externalInputName(audioSlot)} 必须与同编号 ${externalInputName(slot)} 一起使用；当前缺少对应参考视频。`,
                    }));
                }
                return pair;
            });
            content.append(element("div", { className: "h3u-media-grid" }, videoPairs));
            content.append(element("div", { className: "h3u-media-grid" }, AUDIO_SLOTS.slice(0, audioSlotCount).map((slot) => mediaRow(node, stateWidget, state, slot, "audio", scheduleRender))));
        }
        fitNode();
    };
    const scheduleRender = () => {
        cancelAnimationFrame(renderFrame);
        renderFrame = requestAnimationFrame(() => {
            renderFrame = 0;
            render();
        });
    };
    root.addEventListener("paste", async (event) => {
        if (widget(node, "mode")?.value !== "omni_reference") return;
        const files = clipboardMediaFiles(event.clipboardData);
        if (!files.length) return;
        event.preventDefault();
        event.stopPropagation();
        const errors = [];
        let changed = false;
        for (const file of files) {
            const kind = fileKind(file);
            const slot = firstOpenMediaSlot(node, state, kind);
            if (!slot) {
                errors.push(`${kind === "image" ? "图片" : kind === "video" ? "视频" : "音频"}槽位已满：${file.name}`);
                continue;
            }
            try {
                const uploaded = await uploadFile(file);
                if (uploaded.kind !== kind) throw new Error(`媒体类型不匹配：${file.name}`);
                state[slot] = { ...uploaded, trim_start: 0, trim_end: uploaded.duration || 0, use_audio: false };
                changed = true;
            } catch (error) {
                errors.push(`${file.name}：${error.message}`);
            }
        }
        if (changed) {
            setState(node, stateWidget, state);
            scheduleRender();
        }
        if (errors.length) alert(errors.join("\n"));
    });
    const bindPromptSource = () => {
        promptSourceUnsubscribe();
        promptSourceUnsubscribe = subscribePromptSource(node, updateAudioStatus);
    };
    const modeWidget = widget(node, "mode");
    if (modeWidget) {
        const original = modeWidget.callback;
        modeWidget.callback = function () {
            const result = original?.apply(this, arguments);
            requestAnimationFrame(() => {
                removeLegacyModelInputs(node);
                expandCollapsedAutogrowInputs(node);
                pruneModeInputs(node);
                scheduleOriginalInputLabels(node);
                syncDurationControls(node);
                scheduleRender();
            });
            return result;
        };
    }
    const promptWidget = widget(node, "prompt");
    if (promptWidget) {
        const original = promptWidget.callback;
        promptWidget.callback = function () {
            const result = original?.apply(this, arguments);
            updateAudioStatus();
            return result;
        };
    }
    const autoLengthWidget = widget(node, "auto_length_from_audio");
    if (autoLengthWidget) {
        const original = autoLengthWidget.callback;
        autoLengthWidget.callback = function () {
            const result = original?.apply(this, arguments);
            syncDurationControls(node);
            scheduleRender();
            return result;
        };
    }
    const durationWidget = widget(node, "duration");
    if (durationWidget) {
        const original = durationWidget.callback;
        durationWidget.callback = function () {
            const result = original?.apply(this, arguments);
            scheduleRender();
            return result;
        };
    }
    const originalInputAdded = node.onInputAdded;
    node.onInputAdded = function () {
        const result = originalInputAdded?.apply(this, arguments);
        requestAnimationFrame(() => {
            removeLegacyModelInputs(node);
            expandCollapsedAutogrowInputs(node);
            scheduleOriginalInputLabels(node);
        });
        return result;
    };
    const originalConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = originalConnectionsChange?.apply(this, arguments);
        scheduleOriginalInputLabels(node);
        requestAnimationFrame(() => {
            bindPromptSource();
            scheduleRender();
        });
        return result;
    };
    const originalConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        const savedVersion = Number(info?.properties?.minimax_h3_unified_version || 0);
        migrateDurationInput(info);
        migrateWidgets(info, savedVersion);
        const result = originalConfigure?.apply(this, arguments);
        normalizeOutputs(node);
        node.properties = { ...(node.properties || {}), minimax_h3_unified_version: WORKFLOW_SCHEMA_VERSION };
        removeLegacyModelInputs(node);
        expandCollapsedAutogrowInputs(node);
        pruneModeInputs(node);
        scheduleOriginalInputLabels(node);
        restoreState();
        hideWidget(stateWidget);
        bindPromptSource();
        syncDurationControls(node);
        scheduleRender();
        return result;
    };
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(fitNode) : null;
    observer?.observe(content);
    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        observer?.disconnect();
        promptSourceUnsubscribe();
        cancelAnimationFrame(resizeFrame);
        cancelAnimationFrame(renderFrame);
        const labelFrame = INPUT_LABEL_RESTORE.get(node);
        if (labelFrame) cancelAnimationFrame(labelFrame);
        INPUT_LABEL_RESTORE.delete(node);
        const stateFrame = STATE_COMMIT_FRAMES.get(node);
        if (stateFrame) cancelAnimationFrame(stateFrame);
        STATE_COMMIT_FRAMES.delete(node);
        return originalRemoved?.apply(this, arguments);
    };
    bindPromptSource();
    syncDurationControls(node);
    scheduleRender();
}

const STYLE_ID = "minimax-h3-unified-style";
if (!document.getElementById(STYLE_ID)) {
    const stylesheet = document.createElement("link");
    stylesheet.id = STYLE_ID;
    stylesheet.rel = "stylesheet";
    stylesheet.href = new URL("./minimax_h3_unified.css", import.meta.url).href;
    document.head.append(stylesheet);
}

app.registerExtension({
    name: "MiniMax.H3.Unified",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE) return;
        node.properties = { ...(node.properties || {}), minimax_h3_unified_version: WORKFLOW_SCHEMA_VERSION };
        normalizeOutputs(node);
        removeLegacyModelInputs(node);
        scheduleOriginalInputLabels(node);
        requestAnimationFrame(() => {
            removeLegacyModelInputs(node);
            expandCollapsedAutogrowInputs(node);
            pruneModeInputs(node);
            scheduleOriginalInputLabels(node);
        });
        const stateWidget = widget(node, "media_state");
        if (!stateWidget) return;
        hideWidget(stateWidget);
        requestAnimationFrame(() => hideWidget(stateWidget));
        buildPanel(node, stateWidget);
    },
});
