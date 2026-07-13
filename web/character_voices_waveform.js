const WAVEFORM_BUCKETS = 512;
const MAX_CACHE_ENTRIES = 12;
const NORMALIZATION_PERCENTILE = 0.95;
const VISUAL_COMPRESSION = 0.45;
const waveformCache = new Map();

function cacheEnvelope(url, envelope) {
    waveformCache.delete(url);
    waveformCache.set(url, envelope);
    while (waveformCache.size > MAX_CACHE_ENTRIES) {
        waveformCache.delete(waveformCache.keys().next().value);
    }
}

function buildEnvelope(audioBuffer) {
    const channels = Array.from(
        { length: audioBuffer.numberOfChannels },
        (_, index) => audioBuffer.getChannelData(index),
    );
    const sampleCount = audioBuffer.length;
    const envelope = new Float32Array(WAVEFORM_BUCKETS);

    for (let bucket = 0; bucket < WAVEFORM_BUCKETS; bucket += 1) {
        const from = Math.floor((bucket / WAVEFORM_BUCKETS) * sampleCount);
        const to = Math.max(from + 1, Math.floor(((bucket + 1) / WAVEFORM_BUCKETS) * sampleCount));
        const stride = Math.max(1, Math.floor((to - from) / 128));
        let sumSquares = 0;
        let values = 0;

        for (let sample = from; sample < to; sample += stride) {
            for (const channel of channels) {
                const value = channel[sample] || 0;
                sumSquares += value * value;
                values += 1;
            }
        }
        envelope[bucket] = values ? Math.sqrt(sumSquares / values) : 0;
    }

    const nonZero = Array.from(envelope).filter((value) => value > 0).sort((a, b) => a - b);
    // Normalize against a robust peak and compress the visual range so quiet
    // speech remains readable without a single transient flattening the display.
    const reference = nonZero[Math.floor(nonZero.length * NORMALIZATION_PERCENTILE)] || 1;
    for (let index = 0; index < envelope.length; index += 1) {
        envelope[index] = Math.pow(
            Math.min(1, envelope[index] / reference),
            VISUAL_COMPRESSION,
        );
    }
    return envelope;
}

async function decodeEnvelope(url, signal) {
    const cached = waveformCache.get(url);
    if (cached) {
        cacheEnvelope(url, cached);
        return cached;
    }

    const response = await fetch(url, { signal, cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error("Web Audio API is unavailable");

    const context = new AudioContextClass();
    try {
        const audioBuffer = await context.decodeAudioData(await response.arrayBuffer());
        const envelope = buildEnvelope(audioBuffer);
        cacheEnvelope(url, envelope);
        return envelope;
    } finally {
        context.close().catch(() => {});
    }
}

export function createWaveformRenderer(canvas) {
    let envelope = null;
    let requestId = 0;
    let controller = null;
    let playback = { current: 0, start: 0, end: 0, duration: 0 };

    const draw = () => {
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.round(rect.width * pixelRatio);
        const height = Math.round(rect.height * pixelRatio);
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }

        const context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height);
        if (!envelope) return;

        context.save();
        context.scale(pixelRatio, pixelRatio);
        const center = rect.height / 2;
        const barCount = Math.min(envelope.length, Math.max(48, Math.floor(rect.width / 4)));
        const spacing = rect.width / barCount;
        context.lineWidth = Math.min(2.25, Math.max(1.25, spacing * 0.48));
        context.lineCap = "round";

        const drawBars = (strokeStyle, clipFrom = 0, clipTo = rect.width) => {
            context.save();
            context.beginPath();
            context.rect(clipFrom, 0, Math.max(0, clipTo - clipFrom), rect.height);
            context.clip();
            context.strokeStyle = strokeStyle;
            for (let bar = 0; bar < barCount; bar += 1) {
                const from = Math.floor((bar / barCount) * envelope.length);
                const to = Math.max(from + 1, Math.floor(((bar + 1) / barCount) * envelope.length));
                let level = 0;
                for (let index = from; index < to; index += 1) {
                    level = Math.max(level, envelope[index]);
                }
                const halfHeight = Math.max(1, level * (rect.height / 2 - 5));
                const x = (bar + 0.5) * spacing;
                context.beginPath();
                context.moveTo(x, center - halfHeight);
                context.lineTo(x, center + halfHeight);
                context.stroke();
            }
            context.restore();
        };

        const baseGradient = context.createLinearGradient(0, 3, 0, rect.height - 3);
        baseGradient.addColorStop(0, "rgba(125, 211, 252, 0.76)");
        baseGradient.addColorStop(0.5, "rgba(100, 116, 139, 0.52)");
        baseGradient.addColorStop(1, "rgba(125, 211, 252, 0.76)");
        drawBars(baseGradient);

        if (playback.duration > 0) {
            const progressStart = Math.max(0, Math.min(playback.start, playback.duration));
            const progressLimit = Math.max(progressStart, Math.min(playback.end, playback.duration));
            const progressEnd = Math.max(progressStart, Math.min(playback.current, progressLimit));
            if (progressEnd > progressStart) {
                const playedGradient = context.createLinearGradient(0, 2, 0, rect.height - 2);
                playedGradient.addColorStop(0, "rgba(186, 230, 253, 1)");
                playedGradient.addColorStop(0.5, "rgba(56, 189, 248, 0.96)");
                playedGradient.addColorStop(1, "rgba(186, 230, 253, 1)");
                drawBars(
                    playedGradient,
                    (progressStart / playback.duration) * rect.width,
                    (progressEnd / playback.duration) * rect.width,
                );
            }
        }
        context.restore();
    };

    const resizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(draw);
    resizeObserver?.observe(canvas);

    return {
        async load(url) {
            const currentRequest = ++requestId;
            controller?.abort();
            controller = new AbortController();
            envelope = null;
            draw();
            if (!url) return;

            try {
                const decoded = await decodeEnvelope(url, controller.signal);
                if (currentRequest !== requestId) return;
                envelope = decoded;
                draw();
            } catch (error) {
                if (error.name !== "AbortError") {
                    console.warn("Character Voices waveform could not be rendered:", error);
                }
            }
        },
        clear() {
            requestId += 1;
            controller?.abort();
            controller = null;
            envelope = null;
            playback = { current: 0, start: 0, end: 0, duration: 0 };
            draw();
        },
        setPlayback(current, start, end, duration) {
            playback = {
                current: Number.isFinite(current) ? current : 0,
                start: Number.isFinite(start) ? start : 0,
                end: Number.isFinite(end) ? end : 0,
                duration: Number.isFinite(duration) ? duration : 0,
            };
            draw();
        },
        destroy() {
            this.clear();
            resizeObserver?.disconnect();
        },
    };
}
