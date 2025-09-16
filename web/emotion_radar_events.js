/**
 * Emotion Radar Chart Events Module
 * Handles all user interactions and event processing
 */
export class EmotionRadarEvents {
    constructor(core) {
        this.core = core;
        this.lastMousePos = { x: 0, y: 0 };
    }

    setupEventListeners() {
        if (!this.core.canvas) return;

        // Mouse events
        this.core.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.core.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.core.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.core.canvas.addEventListener('mouseleave', this.onMouseLeave.bind(this));

        // Touch events for mobile support
        this.core.canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
        this.core.canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
        this.core.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));

        // Keyboard events
        this.core.canvas.addEventListener('keydown', this.onKeyDown.bind(this));

        // Make canvas focusable for keyboard events
        this.core.canvas.tabIndex = 0;

        // Prevent context menu
        this.core.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }

    getCanvasCoordinates(clientX, clientY) {
        if (!this.core.canvas) return { x: 0, y: 0 };

        const rect = this.core.canvas.getBoundingClientRect();
        const displayX = clientX - rect.left;
        const displayY = clientY - rect.top;

        // Scale coordinates to logical canvas size
        const logicalCanvasWidth = this.core.canvas.width / devicePixelRatio;
        const logicalCanvasHeight = this.core.canvas.height / devicePixelRatio;

        const x = (displayX / rect.width) * logicalCanvasWidth;
        const y = (displayY / rect.height) * logicalCanvasHeight;

        return { x, y };
    }

    onMouseDown(event) {
        event.preventDefault();
        this.core.canvas.focus();

        const { x, y } = this.getCanvasCoordinates(event.clientX, event.clientY);
        this.lastMousePos = { x, y };

        // Check if clicking near center (reset all)
        const distanceFromCenter = Math.sqrt(
            Math.pow(x - this.core.centerX, 2) + Math.pow(y - this.core.centerY, 2)
        );

        if (distanceFromCenter < 15) {
            // Double-click detection for reset
            if (event.detail === 2) {
                this.core.resetAllEmotions();
                return;
            }
        }

        // Check if clicking on an emotion label (for increment)
        const clickedLabel = this.findClickedEmotionLabel(x, y);
        if (clickedLabel) {
            this.incrementEmotion(clickedLabel);
            return;
        }

        // Start dragging
        this.core.isDragging = true;
        this.core.draggedEmotion = this.core.updateEmotionFromPosition(x, y);

        if (this.core.draggedEmotion) {
            this.core.canvas.style.cursor = 'grabbing';
            this.core.visualization.redraw();
        }
    }

    onMouseMove(event) {
        const { x, y } = this.getCanvasCoordinates(event.clientX, event.clientY);
        this.lastMousePos = { x, y };

        if (this.core.isDragging && this.core.draggedEmotion) {
            // Update emotion value during drag
            this.core.updateEmotionFromPosition(x, y);
        } else {
            // Update hover state
            this.updateHoverState(x, y);
        }
    }

    onMouseUp(event) {
        if (this.core.isDragging) {
            this.core.isDragging = false;
            this.core.draggedEmotion = null;
            this.core.canvas.style.cursor = 'default';
            this.core.visualization.redraw();
        }
    }

    onMouseLeave(event) {
        this.core.isDragging = false;
        this.core.draggedEmotion = null;
        this.core.hoveredEmotion = null;
        this.core.canvas.style.cursor = 'default';
        this.core.visualization.redraw();
    }

    // Touch events for mobile support
    onTouchStart(event) {
        event.preventDefault();
        if (event.touches.length === 1) {
            const touch = event.touches[0];
            this.onMouseDown({
                preventDefault: () => {},
                clientX: touch.clientX,
                clientY: touch.clientY,
                detail: 1
            });
        }
    }

    onTouchMove(event) {
        event.preventDefault();
        if (event.touches.length === 1 && this.core.isDragging) {
            const touch = event.touches[0];
            this.onMouseMove({
                clientX: touch.clientX,
                clientY: touch.clientY
            });
        }
    }

    onTouchEnd(event) {
        event.preventDefault();
        this.onMouseUp(event);
    }

    onKeyDown(event) {
        if (!this.core.hoveredEmotion) return;

        const emotion = this.core.hoveredEmotion;
        const currentValue = this.core.emotionValues[emotion.name];
        let newValue = currentValue;

        switch (event.key) {
            case 'ArrowUp':
            case '+':
            case '=':
                newValue = Math.min(1.2, currentValue + 0.1);
                break;
            case 'ArrowDown':
            case '-':
                newValue = Math.max(0.0, currentValue - 0.1);
                break;
            case 'PageUp':
                newValue = Math.min(1.2, currentValue + 0.2);
                break;
            case 'PageDown':
                newValue = Math.max(0.0, currentValue - 0.2);
                break;
            case 'Home':
                newValue = 1.2;
                break;
            case 'End':
            case '0':
                newValue = 0.0;
                break;
            case 'r':
            case 'R':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.core.resetAllEmotions();
                    return;
                }
                break;
            case 'Escape':
                this.core.hoveredEmotion = null;
                this.core.visualization.redraw();
                return;
        }

        if (newValue !== currentValue) {
            event.preventDefault();
            this.core.setEmotionValue(emotion.name, newValue);
        }
    }

    updateHoverState(x, y) {
        // Find nearest emotion for hover effect
        const distanceFromCenter = Math.sqrt(
            Math.pow(x - this.core.centerX, 2) + Math.pow(y - this.core.centerY, 2)
        );

        if (distanceFromCenter > this.core.maxRadius + 30) {
            // Too far from chart
            if (this.core.hoveredEmotion) {
                this.core.hoveredEmotion = null;
                this.core.canvas.style.cursor = 'default';
                this.core.visualization.redraw();
            }
            return;
        }

        const nearestEmotion = this.core.findNearestEmotion(x, y);

        if (nearestEmotion !== this.core.hoveredEmotion) {
            this.core.hoveredEmotion = nearestEmotion;
            this.core.canvas.style.cursor = nearestEmotion ? 'grab' : 'default';
            this.core.visualization.redraw();
        }

        // Check if hovering over an existing emotion handle
        const hoveredHandle = this.findHoveredHandle(x, y);
        if (hoveredHandle) {
            this.core.canvas.style.cursor = 'grab';
        }
    }

    findHoveredHandle(x, y) {
        for (const emotion of this.core.emotions) {
            const value = this.core.emotionValues[emotion.name];
            if (value > 0) {
                const radius = this.core.getRadiusFromEmotionValue(value);
                const point = this.core.polarToCartesian(
                    this.core.centerX, this.core.centerY, radius, emotion.angle
                );

                const distance = Math.sqrt(
                    Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2)
                );

                if (distance < 12) {
                    return emotion;
                }
            }
        }
        return null;
    }

    findClickedEmotionLabel(x, y) {
        // Check if click is near any emotion label
        for (const emotion of this.core.emotions) {
            const labelPoint = this.core.polarToCartesian(
                this.core.centerX, this.core.centerY, this.core.maxRadius + 20, emotion.angle
            );

            const distance = Math.sqrt(
                Math.pow(x - labelPoint.x, 2) + Math.pow(y - labelPoint.y, 2)
            );

            // Check if clicking within label area (roughly 30px radius)
            if (distance < 30) {
                return emotion;
            }
        }
        return null;
    }

    incrementEmotion(emotion) {
        const currentValue = this.core.emotionValues[emotion.name];
        const newValue = Math.min(1.2, currentValue + 0.1);

        this.core.setEmotionValue(emotion.name, newValue);

        // Visual feedback
        this.core.showMessage(`${emotion.name} +0.1 → ${newValue.toFixed(1)}`);
    }

    // Utility for handling wheel events (optional zoom)
    onWheel(event) {
        event.preventDefault();

        // Optional: implement fine-tuning with wheel
        if (this.core.hoveredEmotion) {
            const emotion = this.core.hoveredEmotion;
            const currentValue = this.core.emotionValues[emotion.name];
            const delta = event.deltaY > 0 ? -0.05 : 0.05;
            const newValue = Math.max(0.0, Math.min(1.2, currentValue + delta));

            this.core.setEmotionValue(emotion.name, newValue);
        }
    }
}