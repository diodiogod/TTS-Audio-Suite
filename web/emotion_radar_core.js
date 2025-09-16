import { app } from "../../scripts/app.js";
import { EmotionRadarUI } from "./emotion_radar_ui.js";
import { EmotionRadarEvents } from "./emotion_radar_events.js";
import { EmotionRadarVisualization } from "./emotion_radar_visualization.js";

/**
 * Core Emotion Radar Chart Interface
 * Main class that coordinates all emotion radar chart functionality
 */
export class EmotionRadarInterface {
    constructor(node) {
        this.node = node;
        this.canvas = null;
        this.ctx = null;

        // Emotion configuration
        this.emotions = [
            { name: 'Happy', color: '#FFD700', angle: 0 },
            { name: 'Surprised', color: '#FF69B4', angle: Math.PI / 4 },
            { name: 'Angry', color: '#FF4500', angle: Math.PI / 2 },
            { name: 'Disgusted', color: '#8B4513', angle: 3 * Math.PI / 4 },
            { name: 'Sad', color: '#4169E1', angle: Math.PI },
            { name: 'Afraid', color: '#9370DB', angle: 5 * Math.PI / 4 },
            { name: 'Calm', color: '#20B2AA', angle: 3 * Math.PI / 2 },
            { name: 'Melancholic', color: '#708090', angle: 7 * Math.PI / 4 }
        ];

        // Current emotion values (0.0 to 1.2)
        this.emotionValues = {};
        this.emotions.forEach(emotion => {
            this.emotionValues[emotion.name] = 0.0;
        });

        // Chart configuration
        this.centerX = 140;
        this.centerY = 140;
        this.maxRadius = 110;
        this.chartSize = 280;

        // Interaction state
        this.isDragging = false;
        this.draggedEmotion = null;
        this.hoveredEmotion = null;

        // Color scheme
        this.colors = {
            background: '#1a1a1a',
            grid: '#333333',
            gridText: '#888888',
            chartFill: 'rgba(255, 255, 255, 0.1)',
            chartStroke: '#ffffff',
            centerPoint: '#ffffff'
        };

        // Initialize modules
        this.ui = new EmotionRadarUI(this);
        this.events = new EmotionRadarEvents(this);
        this.visualization = new EmotionRadarVisualization(this);

        this.setupInterface();
    }

    setupInterface() {
        // Create the main interface using UI module
        this.ui.createInterface();

        // Setup event listeners
        this.events.setupEventListeners();

        // Setup canvas resize observer
        this.ui.setupCanvasResize();

        // Sync with existing widget values
        this.syncFromWidgets();

        // Initial render
        this.visualization.redraw();
    }

    syncFromWidgets() {
        // Sync current values from the node's widgets
        this.emotions.forEach(emotion => {
            const widget = this.node.widgets.find(w => w.name === emotion.name);
            if (widget) {
                this.emotionValues[emotion.name] = widget.value || 0.0;
            }
        });
    }

    syncToWidgets() {
        // Update the node's widgets with current radar values
        this.emotions.forEach(emotion => {
            const widget = this.node.widgets.find(w => w.name === emotion.name);
            if (widget) {
                widget.value = this.emotionValues[emotion.name];
                if (widget.callback) {
                    widget.callback(widget.value);
                }
            }
        });

        // Trigger node update
        if (this.node.onResize) {
            this.node.onResize(this.node.size);
        }
    }

    // Coordinate conversion utilities
    polarToCartesian(centerX, centerY, radius, angleInRadians) {
        return {
            x: centerX + (radius * Math.cos(angleInRadians - Math.PI / 2)),
            y: centerY + (radius * Math.sin(angleInRadians - Math.PI / 2))
        };
    }

    cartesianToPolar(x, y) {
        const dx = x - this.centerX;
        const dy = y - this.centerY;
        const radius = Math.sqrt(dx * dx + dy * dy);
        let angle = Math.atan2(dy, dx) + Math.PI / 2;
        if (angle < 0) angle += 2 * Math.PI;
        return { radius, angle };
    }

    getEmotionValueFromRadius(radius) {
        // Convert radius to emotion value (0.0 to 1.2)
        const normalizedRadius = Math.min(radius / this.maxRadius, 1.0);
        return normalizedRadius * 1.2;
    }

    getRadiusFromEmotionValue(value) {
        // Convert emotion value (0.0 to 1.2) to radius
        return (value / 1.2) * this.maxRadius;
    }

    findNearestEmotion(x, y) {
        // Find which emotion axis is closest to the click point
        const { angle } = this.cartesianToPolar(x, y);

        let nearestEmotion = null;
        let minAngleDiff = Math.PI;

        this.emotions.forEach(emotion => {
            let angleDiff = Math.abs(angle - emotion.angle);
            // Handle angle wraparound
            if (angleDiff > Math.PI) {
                angleDiff = 2 * Math.PI - angleDiff;
            }

            if (angleDiff < minAngleDiff) {
                minAngleDiff = angleDiff;
                nearestEmotion = emotion;
            }
        });

        return nearestEmotion;
    }

    setEmotionValue(emotionName, value) {
        // Clamp value to valid range
        value = Math.max(0.0, Math.min(1.2, value));
        this.emotionValues[emotionName] = value;

        // Update visualization
        this.visualization.redraw();

        // Sync to widgets
        this.syncToWidgets();

        // Show value feedback
        this.showMessage(`${emotionName}: ${value.toFixed(2)}`);
    }

    updateEmotionFromPosition(x, y) {
        // Update emotion value based on mouse position
        const nearestEmotion = this.findNearestEmotion(x, y);
        if (!nearestEmotion) return;

        const { radius } = this.cartesianToPolar(x, y);
        const value = this.getEmotionValueFromRadius(radius);

        this.setEmotionValue(nearestEmotion.name, value);
        return nearestEmotion;
    }

    resetAllEmotions() {
        // Reset all emotions to 0.0
        this.emotions.forEach(emotion => {
            this.emotionValues[emotion.name] = 0.0;
        });

        this.visualization.redraw();
        this.syncToWidgets();
        this.showMessage('All emotions reset to 0.0');
    }

    randomizeEmotions() {
        // Set random emotion values for experimentation
        this.emotions.forEach(emotion => {
            this.emotionValues[emotion.name] = Math.random() * 1.2;
        });

        this.visualization.redraw();
        this.syncToWidgets();
        this.showMessage('Emotions randomized');
    }

    // Canvas management
    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.ctx.scale(devicePixelRatio, devicePixelRatio);

        // Update center coordinates based on canvas size
        this.centerX = rect.width / 2;
        this.centerY = rect.height / 2;

        this.visualization.redraw();
    }

    // Message display
    showMessage(message) {
        this.ui.showMessage(message);
    }

    // Export current emotion configuration
    exportEmotionConfig() {
        const config = { ...this.emotionValues };
        const configText = JSON.stringify(config, null, 2);

        navigator.clipboard.writeText(configText).then(() => {
            this.showMessage('Emotion configuration copied to clipboard');
        }).catch(() => {
            alert(`Emotion Configuration:\n${configText}`);
        });
    }
}