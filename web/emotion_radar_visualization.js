/**
 * Emotion Radar Chart Visualization Module
 * Handles all canvas drawing and visual effects for the emotion radar chart
 */
export class EmotionRadarVisualization {
    constructor(core) {
        this.core = core;
        this.animationId = null;
        this.pulseTime = 0;
    }

    redraw() {
        if (!this.core.canvas || !this.core.ctx) return;

        const ctx = this.core.ctx;
        const width = this.core.canvas.width / devicePixelRatio;
        const height = this.core.canvas.height / devicePixelRatio;

        // Clear canvas with gradient background
        this.drawBackground(ctx, width, height);

        // Draw radar grid
        this.drawRadarGrid(ctx);

        // Draw emotion areas (filled)
        this.drawEmotionAreas(ctx);

        // Draw emotion axes and labels
        this.drawEmotionAxes(ctx);

        // Draw center point
        this.drawCenterPoint(ctx);

        // Draw emotion values and handles
        this.drawEmotionHandles(ctx);

        // Draw hover effects
        this.drawHoverEffects(ctx);
    }

    drawBackground(ctx, width, height) {
        // Subtle gradient background
        const gradient = ctx.createRadialGradient(
            this.core.centerX, this.core.centerY, 0,
            this.core.centerX, this.core.centerY, this.core.maxRadius * 1.5
        );
        gradient.addColorStop(0, '#2a2a2a');
        gradient.addColorStop(0.7, '#1e1e1e');
        gradient.addColorStop(1, '#1a1a1a');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
    }

    drawRadarGrid(ctx) {
        const { centerX, centerY, maxRadius } = this.core;

        // Draw concentric circles with subtle styling
        ctx.strokeStyle = this.core.colors.grid;
        ctx.lineWidth = 1;

        // Grid circles at 25%, 50%, 75%, 100%
        const gridLevels = [0.25, 0.5, 0.75, 1.0];
        gridLevels.forEach((level, index) => {
            const radius = maxRadius * level;

            // Make outer circle slightly more prominent
            ctx.globalAlpha = index === gridLevels.length - 1 ? 0.6 : 0.3;

            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.stroke();
        });

        ctx.globalAlpha = 1.0;

        // Draw grid value labels
        this.drawGridLabels(ctx);
    }

    drawGridLabels(ctx) {
        const { centerX, centerY, maxRadius } = this.core;

        ctx.fillStyle = this.core.colors.gridText;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';

        // Draw value labels on the right side
        const levels = [0.3, 0.6, 0.9, 1.2];
        levels.forEach((value, index) => {
            const radius = maxRadius * ((index + 1) / 4);
            const x = centerX + radius + 8;
            const y = centerY + 3;

            ctx.fillText(value.toFixed(1), x, y);
        });
    }

    drawEmotionAreas(ctx) {
        // Draw filled area showing current emotion configuration
        if (this.hasAnyEmotionValues()) {
            ctx.beginPath();
            ctx.fillStyle = 'rgba(100, 200, 255, 0.15)';
            ctx.strokeStyle = 'rgba(100, 200, 255, 0.4)';
            ctx.lineWidth = 2;

            let firstPoint = true;
            this.core.emotions.forEach(emotion => {
                const value = this.core.emotionValues[emotion.name];
                const radius = this.core.getRadiusFromEmotionValue(value);
                const point = this.core.polarToCartesian(
                    this.core.centerX, this.core.centerY, radius, emotion.angle
                );

                if (firstPoint) {
                    ctx.moveTo(point.x, point.y);
                    firstPoint = false;
                } else {
                    ctx.lineTo(point.x, point.y);
                }
            });

            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
    }

    drawEmotionAxes(ctx) {
        const { centerX, centerY, maxRadius } = this.core;

        this.core.emotions.forEach(emotion => {
            const endPoint = this.core.polarToCartesian(centerX, centerY, maxRadius, emotion.angle);

            // Draw axis line
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(endPoint.x, endPoint.y);
            ctx.strokeStyle = emotion.color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.7;
            ctx.stroke();
            ctx.globalAlpha = 1.0;

            // Draw emotion label
            this.drawEmotionLabel(ctx, emotion, endPoint);
        });
    }

    drawEmotionLabel(ctx, emotion, endPoint) {
        const { centerX, centerY, maxRadius } = this.core;

        // Position label slightly outside the chart
        const labelPoint = this.core.polarToCartesian(
            centerX, centerY, maxRadius + 20, emotion.angle
        );

        // Check if this label is being hovered for click feedback
        const isHovered = this.isLabelHovered(labelPoint);

        // Draw clickable background for labels
        if (isHovered) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.beginPath();
            ctx.arc(labelPoint.x, labelPoint.y, 18, 0, 2 * Math.PI);
            ctx.fill();
        }

        ctx.fillStyle = isHovered ? '#ffffff' : emotion.color;
        ctx.font = isHovered ? 'bold 12px Arial' : 'bold 11px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Add subtle text shadow for better readability
        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        ctx.shadowBlur = 2;

        ctx.fillText(emotion.name, labelPoint.x, labelPoint.y);

        // Add small "+" indicator to show it's clickable
        if (isHovered) {
            ctx.font = '8px Arial';
            ctx.fillStyle = '#ffffff';
            ctx.fillText('+0.1', labelPoint.x, labelPoint.y + 12);
        }

        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.shadowBlur = 0;
    }

    isLabelHovered(labelPoint) {
        if (!this.core.events || !this.core.events.lastMousePos) return false;

        const { x, y } = this.core.events.lastMousePos;
        const distance = Math.sqrt(
            Math.pow(x - labelPoint.x, 2) + Math.pow(y - labelPoint.y, 2)
        );

        return distance < 25;
    }

    drawCenterPoint(ctx) {
        const { centerX, centerY } = this.core;

        // Draw elegant center point with glow effect
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 8);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.8)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0.2)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Inner bright center
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 2, 0, 2 * Math.PI);
        ctx.fill();
    }

    drawEmotionHandles(ctx) {
        this.core.emotions.forEach(emotion => {
            const value = this.core.emotionValues[emotion.name];
            if (value > 0) {
                this.drawEmotionHandle(ctx, emotion, value);
            }
        });
    }

    drawEmotionHandle(ctx, emotion, value) {
        const radius = this.core.getRadiusFromEmotionValue(value);
        const point = this.core.polarToCartesian(
            this.core.centerX, this.core.centerY, radius, emotion.angle
        );

        const isHovered = this.core.hoveredEmotion === emotion;
        const isDragged = this.core.draggedEmotion === emotion;

        // Handle size and style based on state
        const handleRadius = isDragged ? 8 : (isHovered ? 7 : 5);
        const glowRadius = isDragged ? 12 : (isHovered ? 10 : 8);

        // Draw glow effect
        const gradient = ctx.createRadialGradient(
            point.x, point.y, 0, point.x, point.y, glowRadius
        );
        gradient.addColorStop(0, emotion.color);
        gradient.addColorStop(0.6, emotion.color + '80');
        gradient.addColorStop(1, emotion.color + '00');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(point.x, point.y, glowRadius, 0, 2 * Math.PI);
        ctx.fill();

        // Draw handle
        ctx.fillStyle = emotion.color;
        ctx.beginPath();
        ctx.arc(point.x, point.y, handleRadius, 0, 2 * Math.PI);
        ctx.fill();

        // Draw inner highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.beginPath();
        ctx.arc(point.x, point.y, handleRadius * 0.5, 0, 2 * Math.PI);
        ctx.fill();

        // Draw value text near handle
        if (isHovered || isDragged) {
            this.drawValueText(ctx, point, value, emotion.color);
        }
    }

    drawValueText(ctx, point, value, color) {
        ctx.fillStyle = color;
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        // Position text above the handle
        const textY = point.y - 15;

        // Add background for better readability
        const text = value.toFixed(2);
        const textWidth = ctx.measureText(text).width;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(point.x - textWidth/2 - 3, textY - 12, textWidth + 6, 14);

        ctx.fillStyle = color;
        ctx.fillText(text, point.x, textY);
    }

    drawHoverEffects(ctx) {
        if (this.core.hoveredEmotion && !this.core.isDragging) {
            const emotion = this.core.hoveredEmotion;

            // Draw subtle axis highlight
            const endPoint = this.core.polarToCartesian(
                this.core.centerX, this.core.centerY, this.core.maxRadius, emotion.angle
            );

            ctx.beginPath();
            ctx.moveTo(this.core.centerX, this.core.centerY);
            ctx.lineTo(endPoint.x, endPoint.y);
            ctx.strokeStyle = emotion.color;
            ctx.lineWidth = 3;
            ctx.globalAlpha = 0.6;
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
    }

    hasAnyEmotionValues() {
        return this.core.emotions.some(emotion =>
            this.core.emotionValues[emotion.name] > 0
        );
    }

    // Animation utilities
    startPulseAnimation() {
        if (this.animationId) return;

        const animate = () => {
            this.pulseTime += 0.05;
            this.redraw();
            this.animationId = requestAnimationFrame(animate);
        };

        animate();
    }

    stopPulseAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
}