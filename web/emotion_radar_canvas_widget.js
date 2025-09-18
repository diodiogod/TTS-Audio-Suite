/**
 * Emotion Radar Chart Canvas Widget
 * Draws directly on ComfyUI canvas - no DOM positioning issues
 */

export function createEmotionRadarCanvasWidget(node) {
    const WIDGET_HEIGHT = 350; // Increased to prevent overflow

    // Emotion configuration
    const emotions = [
        { name: 'Happy', color: '#FFD700', angle: 0 },
        { name: 'Surprised', color: '#FF69B4', angle: Math.PI / 4 },
        { name: 'Angry', color: '#FF4500', angle: Math.PI / 2 },
        { name: 'Disgusted', color: '#8B4513', angle: 3 * Math.PI / 4 },
        { name: 'Sad', color: '#4169E1', angle: Math.PI },
        { name: 'Afraid', color: '#9370DB', angle: 5 * Math.PI / 4 },
        { name: 'Calm', color: '#20B2AA', angle: 3 * Math.PI / 2 },
        { name: 'Melancholic', color: '#708090', angle: 7 * Math.PI / 4 }
    ];

    // Current emotion values
    const emotionValues = {};
    emotions.forEach(emotion => {
        emotionValues[emotion.name] = 0.0;
    });

    // Chart configuration
    const centerX = 160;
    const centerY = 160;
    const maxRadius = 120;

    // Interaction state
    let isDragging = false;
    let clickedEmotion = null; // Track which emotion is currently clicked/active
    let clickedLabel = null; // Track specifically when a label was clicked (for +0.1 display)
    let lastDraggedEmotion = null; // Track last emotion being dragged (for deadzone continuity)
    let centerClickedEmotions = []; // Track emotions affected by center click (for -0.1 display)

    // Center click functionality
    const CENTER_CLICK_RADIUS = 5; // px - very small clickable center area only
    const CENTER_DRAG_DEADZONE = 20; // px - deadzone for dragging

    // Click counter for cumulative reduction display
    let centerClickCount = 0;
    let centerClickTimer = null;
    const CLICK_RESET_DELAY = 1000; // Reset counter after 1 second of no clicks

    // Click counter for individual emotion label clicks
    const emotionClickCounts = {};
    const emotionClickTimers = {};
    emotions.forEach(emotion => {
        emotionClickCounts[emotion.name] = 0;
        emotionClickTimers[emotion.name] = null;
    });

    // Utility functions
    function polarToCartesian(centerX, centerY, radius, angleInRadians) {
        return {
            x: centerX + (radius * Math.cos(angleInRadians - Math.PI / 2)),
            y: centerY + (radius * Math.sin(angleInRadians - Math.PI / 2))
        };
    }

    function cartesianToPolar(x, y) {
        const dx = x - centerX;
        const dy = y - centerY;
        const radius = Math.sqrt(dx * dx + dy * dy);
        let angle = Math.atan2(dy, dx) + Math.PI / 2;
        if (angle < 0) angle += 2 * Math.PI;
        return { radius, angle };
    }

    function getEmotionValueFromRadius(radius) {
        const normalizedRadius = Math.min(radius / maxRadius, 1.0);
        return normalizedRadius * 1.2;
    }

    function getRadiusFromEmotionValue(value) {
        return (value / 1.2) * maxRadius;
    }

    function blendEmotionColors() {
        // Calculate total emotional weight and color contribution
        let totalWeight = 0;
        let r = 0, g = 0, b = 0;

        emotions.forEach(emotion => {
            const value = emotionValues[emotion.name];
            if (value > 0) {
                // Parse emotion color (assumes hex format like "#ff0000")
                const color = emotion.color;
                const hexR = parseInt(color.substring(1, 3), 16);
                const hexG = parseInt(color.substring(3, 5), 16);
                const hexB = parseInt(color.substring(5, 7), 16);

                // Weight by emotion value (higher values contribute more to the blend)
                const weight = value;
                totalWeight += weight;

                r += hexR * weight;
                g += hexG * weight;
                b += hexB * weight;
            }
        });

        if (totalWeight === 0) {
            // No emotions active, return default neutral color
            return { r: 100, g: 200, b: 255 };
        }

        // Calculate weighted average
        r = Math.round(r / totalWeight);
        g = Math.round(g / totalWeight);
        b = Math.round(b / totalWeight);

        // Slightly darken the colors but keep them vibrant (multiply by 0.85 for richer tone)
        r = Math.round(r * 0.85);
        g = Math.round(g * 0.85);
        b = Math.round(b * 0.85);

        return { r, g, b };
    }

    function findNearestEmotion(x, y) {
        const { radius, angle } = cartesianToPolar(x, y);

        // If in drag deadzone, keep using the last dragged emotion
        if (radius <= CENTER_DRAG_DEADZONE && lastDraggedEmotion) {
            return lastDraggedEmotion;
        }

        // Outside deadzone - clear deadzone tracking and use normal selection
        if (radius > CENTER_DRAG_DEADZONE) {
            lastDraggedEmotion = null;
        }

        // Normal angle-based selection outside deadzone
        let nearestEmotion = null;
        let minAngleDiff = Math.PI;

        emotions.forEach(emotion => {
            let angleDiff = Math.abs(angle - emotion.angle);
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

    function isInCenterClickArea(x, y) {
        const distance = Math.sqrt(
            Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2)
        );
        return distance <= CENTER_CLICK_RADIUS;
    }

    function reduceAllEmotions() {
        emotions.forEach(emotion => {
            const currentValue = emotionValues[emotion.name];
            setEmotionValue(emotion.name, Math.max(0.0, currentValue - 0.1));
        });
    }

    function resetAllEmotions() {
        emotions.forEach(emotion => {
            setEmotionValue(emotion.name, 0.0);
        });
    }

    function findClickedEmotionLabel(x, y) {
        for (const emotion of emotions) {
            const labelPoint = polarToCartesian(centerX, centerY, maxRadius + 20, emotion.angle);
            const distance = Math.sqrt(
                Math.pow(x - labelPoint.x, 2) + Math.pow(y - labelPoint.y, 2)
            );
            if (distance < 25) {
                return emotion;
            }
        }
        return null;
    }

    function findHoveredHandle(x, y) {
        for (const emotion of emotions) {
            const value = emotionValues[emotion.name];
            if (value > 0) {
                const radius = getRadiusFromEmotionValue(value);
                const point = polarToCartesian(centerX, centerY, radius, emotion.angle);
                const distance = Math.sqrt(
                    Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2)
                );
                if (distance < 12) { // Slightly larger hover area than visual handle
                    return emotion;
                }
            }
        }
        return null;
    }

    function setEmotionValue(emotionName, value) {
        value = Math.max(0.0, Math.min(1.2, value));
        emotionValues[emotionName] = value;

        // Update corresponding widget
        const widget = node.widgets.find(w => w.name === emotionName);
        if (widget) {
            widget.value = value;
            if (widget.callback) {
                widget.callback(value);
            }
        }
    }

    function updateEmotionFromPosition(x, y) {
        const nearestEmotion = findNearestEmotion(x, y);
        if (!nearestEmotion) return;

        const { radius } = cartesianToPolar(x, y);
        let value;

        // Apply direction projection ONLY in deadzone to prevent bouncing
        if (radius <= CENTER_DRAG_DEADZONE && lastDraggedEmotion === nearestEmotion) {
            // Calculate distance along the emotion's axis from center
            const dx = x - centerX;
            const dy = y - centerY;
            const emotionDx = Math.cos(nearestEmotion.angle - Math.PI / 2);
            const emotionDy = Math.sin(nearestEmotion.angle - Math.PI / 2);

            // Project mouse position onto emotion's axis (dot product)
            const projectedDistance = dx * emotionDx + dy * emotionDy;

            // Only allow positive values (same direction as emotion axis)
            value = projectedDistance > 0 ? getEmotionValueFromRadius(projectedDistance) : 0;
        } else {
            // Normal fluid dragging outside deadzone
            value = getEmotionValueFromRadius(radius);
        }

        setEmotionValue(nearestEmotion.name, value);

        // Track this emotion for deadzone continuity
        lastDraggedEmotion = nearestEmotion;

        // Set glow for the emotion being dragged
        clickedEmotion = nearestEmotion;
        clickedLabel = null; // Clear label indicator during dragging

        return nearestEmotion;
    }

    function syncFromWidgets() {
        emotions.forEach(emotion => {
            const widget = node.widgets.find(w => w.name === emotion.name);
            if (widget) {
                emotionValues[emotion.name] = widget.value || 0.0;
            }
        });
    }

    function importEmotionConfig() {
        // Try to read from clipboard first
        if (navigator.clipboard && navigator.clipboard.readText) {
            navigator.clipboard.readText().then(text => {
                processImportedText(text);
            }).catch(err => {
                console.log("Clipboard read failed, falling back to prompt:", err);
                fallbackImportPrompt();
            });
        } else {
            // Fallback for browsers without clipboard API
            fallbackImportPrompt();
        }
    }

    function fallbackImportPrompt() {
        const importedText = prompt(
            "Paste your emotion configuration (JSON format):\n\n" +
            "Example:\n" +
            '{\n  "Happy": 0.5,\n  "Angry": 0.3,\n  ...\n}'
        );

        if (importedText) {
            processImportedText(importedText);
        }
    }

    function processImportedText(text) {
        try {
            const config = JSON.parse(text.trim());

            // Validate that it's an object
            if (typeof config !== 'object' || config === null) {
                throw new Error("Configuration must be a JSON object");
            }

            let importedCount = 0;
            const validEmotions = emotions.map(e => e.name);

            // Apply valid emotion values
            validEmotions.forEach(emotionName => {
                if (config.hasOwnProperty(emotionName)) {
                    const value = parseFloat(config[emotionName]);
                    if (!isNaN(value)) {
                        const clampedValue = Math.max(0.0, Math.min(1.2, value));
                        setEmotionValue(emotionName, clampedValue);
                        importedCount++;
                    }
                }
            });

            if (importedCount > 0) {
                console.log(`🎭 Successfully imported ${importedCount} emotion values`);
                // Force redraw
                if (node.graph && node.graph.setDirtyCanvas) {
                    node.graph.setDirtyCanvas(true);
                }
            } else {
                alert("No valid emotion values found in the imported configuration.");
            }

        } catch (error) {
            alert(`Import failed: ${error.message}\n\nPlease check that you've copied a valid JSON emotion configuration.`);
        }
    }

    // Drawing functions
    function drawRadarChart(ctx, width, widgetY) {
        const chartY = widgetY + 10;

        // Clear background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, chartY, width, WIDGET_HEIGHT);

        // Save context
        ctx.save();
        ctx.translate(0, chartY);

        // Draw grid circles
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;

        [0.25, 0.5, 0.75, 1.0].forEach((level, index) => {
            const radius = maxRadius * level;
            ctx.globalAlpha = index === 3 ? 0.6 : 0.3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.stroke();
        });

        ctx.globalAlpha = 1.0;

        // Draw emotion axes and labels
        emotions.forEach(emotion => {
            const endPoint = polarToCartesian(centerX, centerY, maxRadius, emotion.angle);

            // Draw axis line
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(endPoint.x, endPoint.y);
            ctx.strokeStyle = emotion.color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.7;
            ctx.stroke();
            ctx.globalAlpha = 1.0;

            // Draw emotion label with click effects
            const labelPoint = polarToCartesian(centerX, centerY, maxRadius + 20, emotion.angle);

            // Draw label with glow effect when emotion is clicked (both dot and label)
            const isEmotionClicked = clickedEmotion === emotion;
            const isCenterClicked = centerClickedEmotions.includes(emotion);
            const hasGlow = isEmotionClicked || isCenterClicked;

            if (hasGlow) {
                // Glow effect - same as dot glow
                ctx.shadowColor = emotion.color;
                ctx.shadowBlur = 15;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            }

            ctx.fillStyle = emotion.color; // Keep original color
            ctx.font = hasGlow ? 'bold 12px Arial' : 'bold 11px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(emotion.name, labelPoint.x, labelPoint.y);

            // Clear shadow for subsequent drawing
            if (hasGlow) {
                ctx.shadowBlur = 0;
            }

            // Add cumulative increase indicator when label was clicked
            if (clickedLabel === emotion && emotionClickCounts[emotion.name] > 0) {
                ctx.font = '9px Arial';
                ctx.fillStyle = '#ffffff';
                const increaseText = `+${(emotionClickCounts[emotion.name] * 0.1).toFixed(1)}`;
                ctx.fillText(increaseText, labelPoint.x, labelPoint.y + 15);
            }

            // Add cumulative reduction indicator when center was clicked
            if (isCenterClicked && centerClickCount > 0) {
                ctx.font = '9px Arial';
                ctx.fillStyle = '#ffffff';
                const reductionText = `-${(centerClickCount * 0.1).toFixed(1)}`;
                ctx.fillText(reductionText, labelPoint.x, labelPoint.y + 15);
            }
        });

        // Draw filled emotion area with blended colors
        if (emotions.some(emotion => emotionValues[emotion.name] > 0)) {
            const blendedColor = blendEmotionColors();

            ctx.beginPath();
            ctx.fillStyle = `rgba(${blendedColor.r}, ${blendedColor.g}, ${blendedColor.b}, 0.25)`;
            ctx.strokeStyle = `rgba(${blendedColor.r}, ${blendedColor.g}, ${blendedColor.b}, 0.8)`;
            ctx.lineWidth = 2;

            let firstPoint = true;
            emotions.forEach(emotion => {
                const value = emotionValues[emotion.name];
                const radius = getRadiusFromEmotionValue(value);
                const point = polarToCartesian(centerX, centerY, radius, emotion.angle);

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

        // Draw emotion handles
        emotions.forEach(emotion => {
            const value = emotionValues[emotion.name];
            if (value > 0) {
                const radius = getRadiusFromEmotionValue(value);
                const point = polarToCartesian(centerX, centerY, radius, emotion.angle);

                const isClicked = clickedEmotion === emotion;
                const isBeingDragged = isDragging && clickedEmotion === emotion;

                // Show glow effect when clicked or dragging
                if (isClicked || isBeingDragged) {
                    const glowRadius = isBeingDragged ? 12 : 10;
                    const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, glowRadius);
                    gradient.addColorStop(0, emotion.color);
                    gradient.addColorStop(0.6, emotion.color + '80');
                    gradient.addColorStop(1, emotion.color + '00');

                    ctx.fillStyle = gradient;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, glowRadius, 0, 2 * Math.PI);
                    ctx.fill();
                }

                // Handle (slightly larger when clicked)
                const handleRadius = isClicked || isBeingDragged ? 6 : 5;
                ctx.fillStyle = emotion.color;
                ctx.beginPath();
                ctx.arc(point.x, point.y, handleRadius, 0, 2 * Math.PI);
                ctx.fill();

                // Inner highlight
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.beginPath();
                ctx.arc(point.x, point.y, handleRadius * 0.4, 0, 2 * Math.PI);
                ctx.fill();
            }
        });

        // Draw center point
        const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 8);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0.2)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 4, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 2, 0, 2 * Math.PI);
        ctx.fill();


        // Draw control buttons
        drawButtons(ctx, width);

        ctx.restore();
    }

    function drawButtons(ctx, width) {
        // Position buttons at bottom with proper spacing
        const buttonY = WIDGET_HEIGHT - 30;
        const buttonWidth = 50;
        const buttonHeight = 20;
        const spacing = 8;

        // Left side buttons
        // Random button
        ctx.fillStyle = '#4ecdc4';
        ctx.fillRect(10, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Random', 10 + buttonWidth/2, buttonY + buttonHeight/2);

        // Reset button
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(10 + buttonWidth + spacing, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Reset', 10 + buttonWidth + spacing + buttonWidth/2, buttonY + buttonHeight/2);

        // Right side buttons (to avoid Sad label overlap)
        const rightButtonX = width - 2*(buttonWidth + spacing) - 10;

        // Export button (right side)
        ctx.fillStyle = '#45b7d1';
        ctx.fillRect(rightButtonX, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Export', rightButtonX + buttonWidth/2, buttonY + buttonHeight/2);

        // Import button (right side)
        ctx.fillStyle = '#9b59b6';
        ctx.fillRect(rightButtonX + buttonWidth + spacing, buttonY, buttonWidth, buttonHeight);
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Import', rightButtonX + buttonWidth + spacing + buttonWidth/2, buttonY + buttonHeight/2);
    }

    // Simple hover tracking - no external DOM events needed

    // Create the widget
    const widget = {
        type: "emotion_radar_canvas",
        name: "emotion_radar_canvas",
        value: "",
        y: 0,
        height: WIDGET_HEIGHT,
        serialize: false,
        isVisible: false,

        computeSize: function(width) {
            return [width || 320, WIDGET_HEIGHT];
        },

        draw: function(ctx, node, widget_width, y, widget_height) {
            this.y = y;
            this.lastWidth = widget_width;
            this.isVisible = true;
            syncFromWidgets();
            drawRadarChart(ctx, widget_width, y);
        },

        mouse: function(event, pos, node) {
            if (!pos) return false;

            const localX = pos[0];
            const localY = pos[1] - this.y - 10; // Adjust for chart offset

            if (localY < 0 || localY > WIDGET_HEIGHT) {
                return false;
            }

            if (event.type === "pointerdown") {
                // Check center click first
                if (isInCenterClickArea(localX, localY)) {
                    // Single click - reduce all emotions by 0.1
                    reduceAllEmotions();

                    // Increment click counter
                    centerClickCount++;

                    // Clear previous timer and set new one
                    if (centerClickTimer) {
                        clearTimeout(centerClickTimer);
                    }

                    // Show cumulative reduction and glow on all emotions
                    centerClickedEmotions = [...emotions];

                    // Reset counter and clear display after delay
                    centerClickTimer = setTimeout(() => {
                        centerClickCount = 0;
                        centerClickedEmotions = [];
                        if (node.graph && node.graph.setDirtyCanvas) {
                            node.graph.setDirtyCanvas(true);
                        }
                    }, CLICK_RESET_DELAY);

                    return true;
                }

                // Check button clicks with new positioning
                const buttonY = WIDGET_HEIGHT - 30;
                const buttonWidth = 50;
                const spacing = 8;
                const rightButtonX = this.lastWidth - 2*(buttonWidth + spacing) - 10;

                if (localY >= buttonY && localY <= buttonY + 20) {
                    if (localX >= 10 && localX <= 10 + buttonWidth) {
                        // Random button (left side)
                        emotions.forEach(emotion => {
                            setEmotionValue(emotion.name, Math.random() * 1.2);
                        });
                        return true;
                    } else if (localX >= 10 + buttonWidth + spacing && localX <= 10 + 2*buttonWidth + spacing) {
                        // Reset button (left side)
                        emotions.forEach(emotion => {
                            setEmotionValue(emotion.name, 0.0);
                        });
                        return true;
                    } else if (localX >= rightButtonX && localX <= rightButtonX + buttonWidth) {
                        // Export button - copy emotion configuration to clipboard
                        const config = {
                            "Happy": emotionValues["Happy"],
                            "Angry": emotionValues["Angry"],
                            "Sad": emotionValues["Sad"],
                            "Surprised": emotionValues["Surprised"],
                            "Afraid": emotionValues["Afraid"],
                            "Disgusted": emotionValues["Disgusted"],
                            "Calm": emotionValues["Calm"],
                            "Melancholic": emotionValues["Melancholic"]
                        };

                        const configText = JSON.stringify(config, null, 2);

                        // Try to copy to clipboard
                        if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(configText).then(() => {
                                console.log("🎭 Emotion configuration copied to clipboard");
                                // Show brief success feedback by temporarily changing button color
                                setTimeout(() => {
                                    // Force a redraw to show feedback
                                    if (node.graph && node.graph.setDirtyCanvas) {
                                        node.graph.setDirtyCanvas(true);
                                    }
                                }, 50);
                            }).catch((err) => {
                                console.error("Failed to copy to clipboard:", err);
                                // Fallback to alert
                                alert(`Emotion Configuration (copied to clipboard failed):\n\n${configText}`);
                            });
                        } else {
                            // Fallback for browsers without clipboard API
                            alert(`Emotion Configuration:\n\n${configText}\n\nCopy this text manually.`);
                        }
                        return true;
                    } else if (localX >= rightButtonX + buttonWidth + spacing && localX <= rightButtonX + 2*buttonWidth + spacing) {
                        // Import button (right side) - load emotion configuration from clipboard
                        importEmotionConfig();
                        return true;
                    }
                }

                // Check label clicks for increment
                const labelClicked = findClickedEmotionLabel(localX, localY);
                if (labelClicked) {
                    clickedEmotion = labelClicked; // Set clicked emotion for glow
                    clickedLabel = labelClicked; // Set clicked label for counter display
                    const currentValue = emotionValues[labelClicked.name];
                    setEmotionValue(labelClicked.name, Math.min(1.2, currentValue + 0.1));

                    // Increment click counter for this emotion
                    emotionClickCounts[labelClicked.name]++;

                    // Clear previous timer for this emotion and set new one
                    if (emotionClickTimers[labelClicked.name]) {
                        clearTimeout(emotionClickTimers[labelClicked.name]);
                    }

                    // Reset counter and clear display after delay
                    emotionClickTimers[labelClicked.name] = setTimeout(() => {
                        emotionClickCounts[labelClicked.name] = 0;
                        if (clickedLabel === labelClicked) {
                            clickedLabel = null;
                        }
                        if (node.setDirtyCanvas) {
                            node.setDirtyCanvas(true);
                        }
                    }, CLICK_RESET_DELAY);

                    return true; // Return true to prevent dragging after label click
                }

                // Check if clicking on a dot/handle
                const handleClicked = findHoveredHandle(localX, localY);
                if (handleClicked) {
                    clickedEmotion = handleClicked; // Set clicked emotion for glow
                    clickedLabel = null; // Don't show +0.1 for dot clicks
                    // Don't return true - allow dragging to continue
                }

                // If clicking elsewhere (not on any emotion), clear the glow
                if (!labelClicked && !handleClicked) {
                    clickedEmotion = null;
                    clickedLabel = null;
                }

                // Start dragging
                isDragging = true;
                updateEmotionFromPosition(localX, localY);
                return true;
            }

            if (event.type === "pointermove") {
                if (isDragging) {
                    updateEmotionFromPosition(localX, localY);
                    return true; // Consume the event when dragging
                }
                return false; // Let other handlers also process move events
            }

            if (event.type === "pointerup") {
                isDragging = false;
                lastDraggedEmotion = null; // Reset deadzone tracking
                // Keep the glow on the emotion after dragging stops (persistent selection)
                return true;
            }

            return false;
        }
    };

    return widget;
}