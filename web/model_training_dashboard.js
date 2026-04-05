import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_CLASS = "UnifiedModelTrainingNode";
const DASHBOARD_MIN_HEIGHT = 292;
const DASHBOARD_MIN_WIDTH = 430;
const DASHBOARD_MIN_NODE_HEIGHT = 430;
const DASHBOARD_BOTTOM_PADDING = 14;
const DASHBOARD_TOP_OFFSET_HINT = 128;
const STALE_RUNNING_MS = 180000;
const POLL_INTERVAL_MS = 400;
const TRAINING_NODES = new Map();
const METRIC_HELP_LINES = [
    "This dashboard is a sanity monitor, not a sound-quality score.",
    "The graph is a rolling recent-loss window, not the full run history. Use it to catch spikes, collapse, or a flat plateau.",
    "A flatter graph usually means learning has slowed down. It does not prove the model is automatically done or that it sounds good.",
    "Best is the lowest generator loss seen so far. It is a useful checkpoint candidate, but the best sounding checkpoint is still decided by listening.",
    "Speed and ETA are rough planning numbers. They get more believable after the first epoch and can drift if machine load changes.",
    "gen is the generator objective, disc is discriminator pressure, mel is spectral reconstruction, kl is regularization, and fm is feature matching.",
    "What to look for: a generally downward or stabilizing trend with fewer violent spikes. What not to do: treat one number as the whole story.",
];
const CHART_HELP_LINES = [
    "Rolling recent-loss window, not full history.",
    "Useful for spotting spikes, collapse, or plateau.",
    "Helpful for sanity checks; listening tests still decide quality.",
];

let pollTimer = null;
let pollInFlight = false;

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function safeNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function formatPhase(value) {
    if (!value) {
        return "Idle";
    }
    return String(value)
        .replace(/_/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatSeconds(value) {
    const total = safeNumber(value, NaN);
    if (!Number.isFinite(total) || total < 0) {
        return "--";
    }
    if (total < 60) {
        return `${Math.round(total)}s`;
    }
    const minutes = Math.floor(total / 60);
    const seconds = Math.round(total % 60);
    if (minutes < 60) {
        return `${minutes}m ${seconds}s`;
    }
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
}

function formatRate(value) {
    const parsed = safeNumber(value, NaN);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        return "--";
    }
    return `${parsed.toFixed(parsed >= 10 ? 1 : 2)} it/s`;
}

function formatLoss(value) {
    const parsed = safeNumber(value, NaN);
    return Number.isFinite(parsed) ? parsed.toFixed(3) : "--";
}

function formatPercent(value) {
    const parsed = clamp(safeNumber(value, 0), 0, 1);
    return `${Math.round(parsed * 100)}%`;
}

function formatStepProgress(currentStep, stepsPerEpoch) {
    const current = Math.max(0, Math.round(safeNumber(currentStep, 0)));
    const total = Math.max(0, Math.round(safeNumber(stepsPerEpoch, 0)));
    return total > 0 ? `${current}/${total} epoch steps` : "--";
}

function formatTotalStepProgress(completedSteps, totalSteps) {
    const current = Math.max(0, Math.round(safeNumber(completedSteps, 0)));
    const total = Math.max(0, Math.round(safeNumber(totalSteps, 0)));
    return total > 0 ? `${current}/${total} total steps` : "--";
}

function getStatusColors(status) {
    switch (status) {
        case "running":
            return { pill: "#1f6f5f", glow: "rgba(53, 196, 162, 0.28)", text: "#d7fff4" };
        case "completed":
            return { pill: "#265c34", glow: "rgba(97, 214, 129, 0.28)", text: "#e5ffea" };
        case "cancelled":
            return { pill: "#5f4b1b", glow: "rgba(232, 187, 77, 0.24)", text: "#fff2c6" };
        case "error":
            return { pill: "#6d2b31", glow: "rgba(255, 105, 97, 0.28)", text: "#ffe7e5" };
        case "starting":
            return { pill: "#705212", glow: "rgba(245, 191, 71, 0.28)", text: "#fff3d5" };
        default:
            return { pill: "#414141", glow: "rgba(160, 160, 160, 0.18)", text: "#f1f1f1" };
    }
}

function ensurePoller() {
    if (pollTimer) {
        return;
    }
    pollTimer = window.setInterval(pollTrainingProgress, POLL_INTERVAL_MS);
}

function stopPollerIfUnused() {
    if (TRAINING_NODES.size === 0 && pollTimer) {
        window.clearInterval(pollTimer);
        pollTimer = null;
    }
}

function setNodeDashboardState(node, state) {
    node.__ttsTrainingDashboardState = deriveDisplayTelemetry(
        stabilizeRunningState(node.__ttsTrainingDashboardState, state)
    );
    requestNodeRedraw(node);
}

function requestNodeRedraw(node) {
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function buildIdleState() {
    return {
        status: "idle",
        phase: "Idle",
        message: "Run the node to see live training progress",
        overall_progress: 0,
        epoch_progress: 0,
        history: [],
    };
}

function isActiveStatus(status) {
    return ["starting", "running"].includes(String(status || "").toLowerCase());
}

function getUpdatedAtSortValue(entry) {
    const raw = String(entry?.updated_at || "").trim();
    if (!raw) {
        return 0;
    }

    const parsed = Date.parse(raw);
    if (Number.isFinite(parsed)) {
        return parsed;
    }

    const digitsOnly = raw.replace(/\D/g, "");
    return Number(digitsOnly.slice(0, 17)) || 0;
}

function withStaleGuard(entry) {
    if (!entry || !isActiveStatus(entry.status)) {
        return entry;
    }

    const updatedAt = getUpdatedAtSortValue(entry);
    if (!updatedAt || (Date.now() - updatedAt) <= STALE_RUNNING_MS) {
        return entry;
    }

    return {
        ...entry,
        status: "error",
        phase: "stale",
        error: entry.error || "Training session stopped updating.",
    };
}

function getJobFingerprint(entry) {
    return String(
        entry?.progress_file ||
        entry?.job_dir ||
        entry?.model_path ||
        entry?.model_name ||
        ""
    );
}

function deriveDisplayTelemetry(entry) {
    const state = { ...(entry || {}) };
    const startedAt = safeNumber(state.started_at, NaN);
    const totalEpochs = Math.max(1, Math.round(safeNumber(state.total_epochs, 0)));
    const stepsPerEpoch = Math.max(1, Math.round(safeNumber(state.steps_per_epoch, 0)));
    const totalSteps = Math.max(1, Math.round(safeNumber(state.total_steps, totalEpochs * stepsPerEpoch)));
    const completedTotalSteps = clamp(
        Math.floor(safeNumber(state.completed_total_steps, safeNumber(state.overall_progress, 0) * totalSteps)),
        0,
        totalSteps
    );
    const overallProgress = clamp(completedTotalSteps / totalSteps, 0, 1);

    state.display_eta_sec = state.eta_sec;
    state.display_it_per_sec = state.it_per_sec;
    state.display_total_steps = totalSteps;
    state.display_completed_total_steps = completedTotalSteps;
    state.display_overall_progress = overallProgress;
    const currentEpoch = Math.max(0, Math.round(safeNumber(state.current_epoch, 0)));
    const epochBaseSteps = Math.max(0, (Math.max(currentEpoch, 1) - 1) * stepsPerEpoch);
    state.display_current_step = clamp(completedTotalSteps - epochBaseSteps, 0, stepsPerEpoch);

    if (!isActiveStatus(state.status) || !Number.isFinite(startedAt) || overallProgress <= 0.005 || overallProgress >= 1) {
        return state;
    }

    const elapsed = Math.max((Date.now() / 1000) - startedAt, 1e-6);
    const estimatedTotalDuration = elapsed / overallProgress;
    const derivedEtaSec = Math.max(estimatedTotalDuration - elapsed, 0);
    const derivedItPerSec = completedTotalSteps / elapsed;

    if (Number.isFinite(derivedEtaSec)) {
        state.display_eta_sec = derivedEtaSec;
    }
    if (Number.isFinite(derivedItPerSec) && derivedItPerSec > 0) {
        state.display_it_per_sec = derivedItPerSec;
    }

    return state;
}

function stabilizeRunningState(previousEntry, nextEntry) {
    if (!previousEntry || !nextEntry) {
        return nextEntry;
    }

    const previous = { ...previousEntry };
    const next = { ...nextEntry };
    if (!isActiveStatus(previous.status) || !isActiveStatus(next.status)) {
        return next;
    }
    if (getJobFingerprint(previous) !== getJobFingerprint(next)) {
        return next;
    }

    const totalEpochs = Math.max(1, Math.round(safeNumber(next.total_epochs, previous.total_epochs || 0)));
    const stepsPerEpoch = Math.max(1, Math.round(safeNumber(next.steps_per_epoch, previous.steps_per_epoch || 0)));
    const totalSteps = Math.max(1, Math.round(safeNumber(next.total_steps, totalEpochs * stepsPerEpoch)));

    const previousCompletedTotalSteps = clamp(
        Math.floor(safeNumber(
            previous.display_completed_total_steps ?? previous.completed_total_steps,
            safeNumber(previous.overall_progress, 0) * totalSteps
        )),
        0,
        totalSteps
    );
    const nextCompletedTotalSteps = clamp(
        Math.floor(safeNumber(
            next.completed_total_steps,
            safeNumber(next.overall_progress, 0) * totalSteps
        )),
        0,
        totalSteps
    );
    const allowedStepRegression = 3;
    if (nextCompletedTotalSteps < previousCompletedTotalSteps &&
        (previousCompletedTotalSteps - nextCompletedTotalSteps) <= allowedStepRegression) {
        next.completed_total_steps = previousCompletedTotalSteps;
        next.overall_progress = previousCompletedTotalSteps / totalSteps;
    }

    const previousEpoch = Math.max(0, Math.round(safeNumber(previous.current_epoch, 0)));
    const nextEpoch = Math.max(0, Math.round(safeNumber(next.current_epoch, 0)));
    const previousStep = Math.max(0, Math.round(safeNumber(previous.current_step, 0)));
    const nextStep = Math.max(0, Math.round(safeNumber(next.current_step, 0)));
    const previousEpochProgress = clamp(safeNumber(previous.epoch_progress, 0), 0, 1);
    const nextEpochProgress = clamp(safeNumber(next.epoch_progress, 0), 0, 1);

    if (nextEpoch < previousEpoch && (previousCompletedTotalSteps - nextCompletedTotalSteps) <= allowedStepRegression) {
        next.current_epoch = previousEpoch;
        next.current_step = previousStep;
        next.epoch_progress = previousEpochProgress;
        return next;
    }

    if (nextEpoch === previousEpoch &&
        nextStep < previousStep &&
        (previousStep - nextStep) <= allowedStepRegression) {
        next.current_step = previousStep;
        next.epoch_progress = Math.max(previousEpochProgress, nextEpochProgress);
    }

    return next;
}

function resolveDashboardState(nodes, nodeId) {
    if (nodes[nodeId]) {
        return withStaleGuard(nodes[nodeId]);
    }

    const entries = Object.values(nodes || {});
    const liveEntries = entries.filter((entry) => {
        const status = String(entry?.status || "");
        return ["starting", "running"].includes(status);
    });
    const terminalEntries = entries.filter((entry) => {
        const status = String(entry?.status || "");
        return ["completed", "error", "cancelled"].includes(status);
    });

    const sortByUpdatedAt = (left, right) => {
        const leftValue = getUpdatedAtSortValue(left);
        const rightValue = getUpdatedAtSortValue(right);
        return rightValue - leftValue;
    };

    if (liveEntries.length > 0) {
        liveEntries.sort(sortByUpdatedAt);
        return withStaleGuard(liveEntries[0]);
    }

    if (terminalEntries.length > 0) {
        terminalEntries.sort(sortByUpdatedAt);
        return terminalEntries[0];
    }

    if (entries.length > 0) {
        entries.sort(sortByUpdatedAt);
        return withStaleGuard(entries[0]);
    }

    return buildIdleState();
}

function fetchTrainingProgressWithTimeout(timeoutMs = 2500) {
    return Promise.race([
        api.fetchApi("/api/tts-audio-suite/training-progress"),
        new Promise((_, reject) => {
            window.setTimeout(() => reject(new Error("Training dashboard poll timed out")), timeoutMs);
        }),
    ]);
}

async function pollTrainingProgress() {
    if (pollInFlight || TRAINING_NODES.size === 0) {
        return;
    }

    pollInFlight = true;
    try {
        const response = await fetchTrainingProgressWithTimeout();
        if (!response.ok) {
            throw new Error(`Training progress request failed: ${response.status}`);
        }

        const payload = await response.json();
        const nodes = payload?.nodes || {};
        for (const [nodeId, node] of TRAINING_NODES) {
            setNodeDashboardState(node, resolveDashboardState(nodes, nodeId));
        }
    } catch (error) {
        for (const [, node] of TRAINING_NODES) {
            const previous = node.__ttsTrainingDashboardState || buildIdleState();
            setNodeDashboardState(node, {
                ...previous,
                status: previous.status === "running" ? previous.status : "idle",
                phase: previous.phase || "Idle",
                message: `Dashboard unavailable: ${error.message}`,
            });
        }
    } finally {
        pollInFlight = false;
    }
}

function drawWrappedText(ctx, text, x, y, maxWidth, lineHeight, maxLines = 3) {
    const lines = wrapTextLines(ctx, text, maxWidth, maxLines);
    lines.forEach((line, index) => {
        ctx.fillText(line, x, y + index * lineHeight);
    });
}

function wrapTextLines(ctx, text, maxWidth, maxLines = Infinity) {
    const words = String(text || "").split(/\s+/).filter(Boolean);
    const lines = [];
    let currentLine = "";

    for (const word of words) {
        const candidate = currentLine ? `${currentLine} ${word}` : word;
        if (ctx.measureText(candidate).width <= maxWidth || !currentLine) {
            currentLine = candidate;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
        if (lines.length >= maxLines) {
            break;
        }
    }

    if (currentLine && lines.length < maxLines) {
        lines.push(currentLine);
    }

    return lines.slice(0, maxLines);
}

function drawRoundedRect(ctx, x, y, width, height, radius, fillStyle) {
    ctx.save();
    ctx.fillStyle = fillStyle;
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, radius);
    ctx.fill();
    ctx.restore();
}

function pointInRect(x, y, rect) {
    if (!rect) {
        return false;
    }
    return x >= rect.x && x <= (rect.x + rect.width) && y >= rect.y && y <= (rect.y + rect.height);
}

function drawHelpBadge(ctx, x, y, active) {
    drawRoundedRect(ctx, x, y, 18, 18, 9, active ? "rgba(98, 221, 196, 0.24)" : "rgba(255,255,255,0.08)");
    ctx.save();
    ctx.fillStyle = active ? "#b6fff0" : "#b9c9c7";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("?", x + 9, y + 9);
    ctx.restore();
}

function drawTextTooltip(ctx, x, y, width, title, lines) {
    ctx.save();
    ctx.font = "10px sans-serif";
    const wrappedLines = [];
    const contentWidth = width - 24;
    for (const line of lines) {
        wrappedLines.push(...wrapTextLines(ctx, line, contentWidth));
    }
    ctx.restore();

    const tooltipHeight = 36 + wrappedLines.length * 12 + 10;
    drawRoundedRect(ctx, x, y, width, tooltipHeight, 12, "rgba(9, 16, 16, 0.96)");
    ctx.save();
    ctx.strokeStyle = "rgba(141, 224, 204, 0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x + 0.5, y + 0.5, width - 1, tooltipHeight - 1, 12);
    ctx.stroke();

    ctx.fillStyle = "#ecfffb";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(title, x + 12, y + 16);

    ctx.fillStyle = "#a9bfbc";
    ctx.font = "10px sans-serif";
    wrappedLines.forEach((line, index) => {
        ctx.fillText(line, x + 12, y + 34 + index * 12);
    });
    ctx.restore();
    return { x, y, width, height: tooltipHeight };
}

function drawHelpTooltip(ctx, x, y, width) {
    return drawTextTooltip(ctx, x, y, width, "How to read this", METRIC_HELP_LINES);
}

function drawChartTooltip(ctx, x, y, width) {
    return drawTextTooltip(ctx, x, y, width, "Recent loss trend", CHART_HELP_LINES);
}

function drawSparkline(ctx, x, y, width, height, trace, history, currentApproxLoss) {
    const traceValues = (trace || [])
        .map((entry) => safeNumber(entry.total_loss, NaN))
        .filter((value) => Number.isFinite(value));
    const epochValues = (history || [])
        .map((entry) => safeNumber(entry.total_loss, NaN))
        .filter((value) => Number.isFinite(value));

    const values = traceValues.length >= 2 ? traceValues : epochValues;
    if (values.length < 2 && Number.isFinite(currentApproxLoss)) {
        values.push(currentApproxLoss);
    }

    drawRoundedRect(ctx, x, y, width, height, 12, "rgba(255,255,255,0.04)");

    if (values.length < 2) {
        ctx.save();
        ctx.fillStyle = "#8b8b8b";
        ctx.font = "12px sans-serif";
        ctx.fillText("Loss curve appears after a couple of updates", x + 14, y + height / 2);
        ctx.restore();
        return;
    }

    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const currentValue = values[values.length - 1];
    const spread = Math.max(maxValue - minValue, 1e-6);
    const chartLeft = x + 14;
    const chartRight = x + width - 14;
    const chartTop = y + 34;
    const chartBottom = y + height - 10;
    const chartHeight = chartBottom - chartTop;

    ctx.save();

    for (let i = 0; i < 3; i += 1) {
        const gridY = chartTop + (chartHeight * i) / 2;
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(chartLeft, gridY);
        ctx.lineTo(chartRight, gridY);
        ctx.stroke();
    }

    const points = values.map((value, index) => {
        const px = chartLeft + (index / Math.max(values.length - 1, 1)) * (chartRight - chartLeft);
        const py = chartBottom - ((value - minValue) / spread) * chartHeight;
        return { x: px, y: py };
    });

    ctx.beginPath();
    points.forEach((point, index) => {
        if (index === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });
    ctx.lineTo(chartRight, chartBottom);
    ctx.lineTo(chartLeft, chartBottom);
    ctx.closePath();
    ctx.fillStyle = "rgba(90, 208, 179, 0.12)";
    ctx.fill();

    ctx.strokeStyle = "rgba(148, 224, 203, 0.95)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((point, index) => {
        if (index === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    });
    ctx.stroke();

    const lastPoint = points[points.length - 1];
    ctx.fillStyle = "#96eed9";
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, 3.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#9fc5bc";
    ctx.font = "10px sans-serif";
    ctx.fillText("Recent loss trend", chartLeft, y + 15);
    ctx.fillStyle = "#6f8480";
    ctx.font = "9px sans-serif";
    ctx.fillText(
        traceValues.length >= 2 ? `rolling window (${values.length} updates)` : "epoch-summary fallback",
        chartLeft,
        y + 25
    );

    ctx.textAlign = "right";
    ctx.fillStyle = "#89e3cd";
    ctx.fillText(`now ${currentValue.toFixed(1)}`, chartRight, y + 15);
    ctx.fillStyle = "#7f8d8b";
    ctx.fillText(`range ${minValue.toFixed(1)} - ${maxValue.toFixed(1)}`, chartRight, y + height - 2);
    ctx.textAlign = "left";
    ctx.restore();
    return { x, y, width, height };
}

function drawMetricChip(ctx, x, y, width, title, value) {
    drawRoundedRect(ctx, x, y, width, 38, 8, "rgba(255,255,255,0.06)");
    ctx.save();
    ctx.fillStyle = "#8ea1a0";
    ctx.font = "10px sans-serif";
    ctx.fillText(title, x + 10, y + 13);
    ctx.fillStyle = "#f4f7f7";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(value, x + 10, y + 29);
    ctx.restore();
}

function getDashboardHeight(node, y = 0) {
    const nodeHeight = safeNumber(node?.size?.[1], DASHBOARD_MIN_NODE_HEIGHT);
    const availableHeight = Math.floor(nodeHeight - y - DASHBOARD_BOTTOM_PADDING);
    return Math.max(DASHBOARD_MIN_HEIGHT, availableHeight);
}

function resizeDashboardWidgetIfNeeded(widget, node, y = 0) {
    const desiredHeight = getDashboardHeight(node, y);
    if (Math.abs(safeNumber(widget.height, 0) - desiredHeight) < 1) {
        return desiredHeight;
    }

    widget.height = desiredHeight;
    widget.computedHeight = desiredHeight;
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
    return desiredHeight;
}

function drawDashboard(ctx, node, widgetWidth, y, dashboardHeight, widget) {
    const state = node.__ttsTrainingDashboardState || buildIdleState();
    const status = state.status || "idle";
    const colors = getStatusColors(status);

    const innerX = 12;
    const innerY = y + 10;
    const innerWidth = widgetWidth - 24;

    drawRoundedRect(ctx, innerX, innerY, innerWidth, dashboardHeight - 20, 14, "#151819");
    drawRoundedRect(ctx, innerX, innerY, innerWidth, 42, 14, colors.glow);

    ctx.save();
    ctx.fillStyle = "#f7f7f5";
    ctx.font = "bold 15px sans-serif";
    ctx.fillText("Training Dashboard", innerX + 14, innerY + 19);

    ctx.fillStyle = "#9fb0af";
    ctx.font = "11px sans-serif";
    const engineLabel = state.engine_type ? String(state.engine_type).toUpperCase() : "No Active Job";
    ctx.fillText(engineLabel, innerX + 14, innerY + 35);

    const pillText = (status || "idle").toUpperCase();
    ctx.font = "bold 11px sans-serif";
    const pillWidth = Math.max(62, ctx.measureText(pillText).width + 18);
    const helpBadgeX = innerX + innerWidth - pillWidth - 38;
    const helpBadgeY = innerY + 12;
    drawRoundedRect(ctx, innerX + innerWidth - pillWidth - 14, innerY + 10, pillWidth, 22, 11, colors.pill);
    ctx.fillStyle = colors.text;
    ctx.fillText(pillText, innerX + innerWidth - pillWidth - 5, innerY + 25);
    drawHelpBadge(ctx, helpBadgeX, helpBadgeY, Boolean(widget?.tooltipPinned || widget?.helpHovered));
    ctx.restore();

    if (widget) {
        widget.helpRect = { x: helpBadgeX, y: helpBadgeY, width: 18, height: 18 };
        widget.tooltipRect = null;
        widget.chartRect = null;
    }

    if (status === "idle") {
        ctx.save();
        ctx.fillStyle = "#a3adad";
        ctx.font = "13px sans-serif";
        drawWrappedText(
            ctx,
            state.message || "Run the node to see live training progress",
            innerX + 16,
            innerY + 72,
            innerWidth - 32,
            17,
            3
        );
        ctx.fillStyle = "#667171";
        ctx.font = "11px sans-serif";
        drawWrappedText(
            ctx,
            "The widget updates live while the backend writes training metrics.",
            innerX + 16,
            innerY + 112,
            innerWidth - 32,
            15,
            3
        );
        ctx.restore();
        if (widget?.tooltipPinned || widget?.helpHovered) {
            const tooltipWidth = Math.min(320, innerWidth - 28);
            widget.tooltipRect = drawHelpTooltip(ctx, innerX + innerWidth - tooltipWidth - 14, innerY + 48, tooltipWidth);
        }
        return;
    }

    const modelName = state.model_name || "Unnamed model";
    const phase = formatPhase(state.phase);
    const epoch = safeNumber(state.current_epoch, 0);
    const totalEpochs = safeNumber(state.total_epochs, 0);
    const currentStep = safeNumber(state.display_current_step, safeNumber(state.current_step, 0));
    const stepsPerEpoch = safeNumber(state.steps_per_epoch, 0);
    const completedTotalSteps = safeNumber(state.display_completed_total_steps, safeNumber(state.completed_total_steps, 0));
    const totalSteps = safeNumber(state.display_total_steps, safeNumber(state.total_steps, 0));
    const overallProgress = clamp(safeNumber(state.display_overall_progress, safeNumber(state.overall_progress, 0)), 0, 1);
    const currentMetrics = state.current_metrics || {};
    const approxLoss = Number.isFinite(safeNumber(currentMetrics.loss_gen_all, NaN))
        ? safeNumber(currentMetrics.loss_gen_all, 0) + safeNumber(currentMetrics.loss_disc_all, 0)
        : NaN;

    ctx.save();
    ctx.fillStyle = "#f0f2f1";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(modelName, innerX + 14, innerY + 62);

    ctx.fillStyle = "#90a1a1";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${phase}${state.sample_rate ? `  |  ${state.sample_rate}` : ""}`, innerX + 14, innerY + 79);

    drawRoundedRect(ctx, innerX + 14, innerY + 90, innerWidth - 28, 12, 6, "rgba(255,255,255,0.08)");
    drawRoundedRect(ctx, innerX + 14, innerY + 90, (innerWidth - 28) * overallProgress, 12, 6, "#5ad0b3");

    ctx.fillStyle = "#d6ece7";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(
        `Epoch ${epoch}${totalEpochs ? ` / ${totalEpochs}` : ""}  |  ${formatTotalStepProgress(completedTotalSteps, totalSteps)}  |  ${formatPercent(overallProgress)}`,
        innerX + 14,
        innerY + 118
    );
    ctx.restore();

    const chipY = innerY + 130;
    const chipWidth = (innerWidth - 42) / 3;
    drawMetricChip(ctx, innerX + 14, chipY, chipWidth, "Speed", formatRate(state.display_it_per_sec ?? state.it_per_sec));
    drawMetricChip(ctx, innerX + 22 + chipWidth, chipY, chipWidth, "ETA", formatSeconds(state.display_eta_sec ?? state.eta_sec));
    drawMetricChip(
        ctx,
        innerX + 30 + chipWidth * 2,
        chipY,
        chipWidth,
        "Best",
        formatLoss(state.best_gen_loss)
    );

    const chartY = innerY + 176;
    const footerReservedHeight = status === "error" ? 42 : 34;
    const footerY = innerY + dashboardHeight - footerReservedHeight;
    const chartHeight = Math.max(56, footerY - chartY - 12);
    const chartRect = drawSparkline(
        ctx,
        innerX + 14,
        chartY,
        innerWidth - 28,
        chartHeight,
        state.recent_loss_trace,
        state.history,
        approxLoss
    );
    if (widget) {
        widget.chartRect = chartRect;
    }

    const footerMetrics = [
        formatStepProgress(currentStep, stepsPerEpoch),
        `gen ${formatLoss(currentMetrics.loss_gen_all)}`,
        `disc ${formatLoss(currentMetrics.loss_disc_all)}`,
        `mel ${formatLoss(currentMetrics.loss_mel)}`,
        `kl ${formatLoss(currentMetrics.loss_kl)}`,
        `fm ${formatLoss(currentMetrics.loss_fm)}`,
    ];
    if (state.last_epoch?.epoch_time_sec) {
        footerMetrics.push(`epoch ${formatSeconds(state.last_epoch.epoch_time_sec)}`);
    }
    if (state.clip_count) {
        footerMetrics.push(`${state.clip_count} clips`);
    }

    ctx.save();
    ctx.fillStyle = status === "error" ? "#ffd6d1" : "#aab8b8";
    ctx.font = "11px sans-serif";
    const footerText = status === "error" && state.error
        ? `Error: ${state.error}`
        : footerMetrics.join("  •  ");
    drawWrappedText(ctx, footerText.slice(0, 180), innerX + 14, footerY, innerWidth - 28, 14, 2);
    ctx.restore();

    if (widget?.tooltipPinned || widget?.helpHovered) {
        const tooltipWidth = Math.min(320, innerWidth - 28);
        widget.tooltipRect = drawHelpTooltip(ctx, innerX + innerWidth - tooltipWidth - 14, innerY + 48, tooltipWidth);
    } else if (widget?.chartHovered) {
        const tooltipWidth = Math.min(280, innerWidth - 28);
        widget.tooltipRect = drawChartTooltip(ctx, innerX + innerWidth - tooltipWidth - 14, chartY + 10, tooltipWidth);
    }
}

function createDashboardWidget(node) {
    return {
        type: "tts_training_dashboard",
        name: "tts_training_dashboard",
        value: "",
        height: DASHBOARD_MIN_HEIGHT,
        serialize: false,
        helpHovered: false,
        tooltipPinned: false,
        helpRect: null,
        tooltipRect: null,
        chartHovered: false,
        chartRect: null,
        computeSize(width) {
            return [width || 360, Math.max(DASHBOARD_MIN_HEIGHT, safeNumber(this.height, DASHBOARD_MIN_HEIGHT))];
        },
        draw(ctx, currentNode, widgetWidth, y, widgetHeight) {
            this.last_y = y;
            const height = resizeDashboardWidgetIfNeeded(this, currentNode, y);
            drawDashboard(ctx, currentNode, widgetWidth, y, Math.max(safeNumber(widgetHeight, 0), height), this);
        },
        mouse(event, pos, currentNode) {
            if (!pos) {
                return false;
            }

            const insideHelp = pointInRect(pos[0], pos[1], this.helpRect);
            const insideTooltip = pointInRect(pos[0], pos[1], this.tooltipRect);
            const insideChart = pointInRect(pos[0], pos[1], this.chartRect);

            if (event?.type === "pointerdown") {
                if (insideHelp) {
                    this.tooltipPinned = !this.tooltipPinned;
                    this.helpHovered = insideHelp;
                    requestNodeRedraw(currentNode || node);
                    return true;
                }
                if (this.tooltipPinned && !insideTooltip) {
                    this.tooltipPinned = false;
                    requestNodeRedraw(currentNode || node);
                }
                return insideTooltip;
            }

            if (event?.type === "pointermove" || event?.type === "mousemove") {
                if (this.helpHovered !== insideHelp) {
                    this.helpHovered = insideHelp;
                    requestNodeRedraw(currentNode || node);
                }
                if (!this.tooltipPinned && this.chartHovered !== insideChart) {
                    this.chartHovered = insideChart;
                    requestNodeRedraw(currentNode || node);
                }
            }

            if (event?.type === "pointerleave" || event?.type === "mouseleave") {
                let needsRedraw = false;
                if (this.helpHovered) {
                    this.helpHovered = false;
                    needsRedraw = true;
                }
                if (this.chartHovered) {
                    this.chartHovered = false;
                    needsRedraw = true;
                }
                if (needsRedraw) {
                    requestNodeRedraw(currentNode || node);
                }
            }

            return insideHelp || insideChart || (this.tooltipPinned && insideTooltip);
        },
    };
}

function registerNode(node) {
    const key = String(node.id);
    TRAINING_NODES.set(key, node);
    if (!node.__ttsTrainingDashboardState) {
        setNodeDashboardState(node, buildIdleState());
    }
    ensurePoller();
    pollTrainingProgress();
}

api.addEventListener("executing", () => {
    pollTrainingProgress();
});

api.addEventListener("executed", () => {
    pollTrainingProgress();
});

app.registerExtension({
    name: "TTS_Audio_Suite.ModelTrainingDashboard",

    async setup() {
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function() {
            const prompt = await originalGraphToPrompt.apply(this, arguments);

            if (prompt?.output) {
                for (const nodeId in prompt.output) {
                    const nodeData = prompt.output[nodeId];
                    if (nodeData.class_type === TARGET_CLASS) {
                        nodeData.inputs.node_id = nodeId;
                    }
                }
            }

            return prompt;
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_CLASS) {
            return;
        }

        const originalNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalNodeCreated ? originalNodeCreated.apply(this, arguments) : undefined;

            try {
                if (!this.widgets) {
                    this.widgets = [];
                }
                const widget = createDashboardWidget(this);
                this.widgets.push(widget);
                this.setSize([
                    Math.max(this.size?.[0] || 380, DASHBOARD_MIN_WIDTH),
                    Math.max(this.size?.[1] || 420, DASHBOARD_MIN_NODE_HEIGHT),
                ]);
                registerNode(this);
            } catch (error) {
                console.error("TTS Audio Suite: Failed to create training dashboard widget:", error);
            }

            return result;
        };

        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            TRAINING_NODES.delete(String(this.id));
            stopPollerIfUnused();
            return originalOnRemoved ? originalOnRemoved.apply(this, arguments) : undefined;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            this.setSize([
                Math.max(this.size?.[0] || 380, DASHBOARD_MIN_WIDTH),
                Math.max(this.size?.[1] || 420, DASHBOARD_MIN_NODE_HEIGHT),
            ]);
            const widget = this.widgets?.find((item) => item?.name === "tts_training_dashboard");
            if (widget) {
                resizeDashboardWidgetIfNeeded(widget, this, safeNumber(widget.last_y, DASHBOARD_TOP_OFFSET_HINT));
            }
            return result;
        };

        const originalOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function(size) {
            const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
            const widget = this.widgets?.find((item) => item?.name === "tts_training_dashboard");
            if (widget) {
                resizeDashboardWidgetIfNeeded(widget, this, safeNumber(widget.last_y, DASHBOARD_TOP_OFFSET_HINT));
            }
            return result;
        };
    },
});
