/**
 * Audio Wave Analyzer Widget Management Module
 * Handles widget creation, positioning, and spacer management
 */
export class AudioAnalyzerWidgets {
    constructor(core) {
        this.core = core;
        this.interfaceHeight = 420;
        this.bottomSpacerHeight = 16;
    }

    setWidgetHeightSafe(widget, height) {
        if (!widget) return;
        try {
            widget.height = height;
        } catch {
            // Newer ComfyUI BaseWidget exposes height as getter-only.
        }
        try {
            widget.computedHeight = height;
        } catch {
            // Ignore readonly implementations.
        }
    }

    setupWidgetHeight(widget, height = this.interfaceHeight) {
        // Ensure the widget properly reserves its height in ComfyUI's layout system
        const interfaceHeight = height;

        // Set the widget to properly reserve space in the layout
        this.setWidgetHeightSafe(widget, interfaceHeight);

        // Override ComfyUI's height calculation methods
        widget.computeSize = function(width) {
            return [width || 760, interfaceHeight]; // Reserve full height in layout
        };

        widget.getHeight = function() {
            return interfaceHeight;
        };

        // Simple draw method that doesn't interfere with positioning
        widget.draw = function(ctx, node, widget_width, y, widget_height) {
            ctx.fillStyle = 'rgba(42, 42, 42, 0.35)';
            ctx.fillRect(0, y, widget_width, Math.min(widget_height, interfaceHeight));
        };

        // Override mouse handling to ensure proper event coordination
        widget.mouse = function(event, pos, node) {
            // Let the interface handle its own mouse events
            return false; // Don't consume the event
        };

        // Ensure the DOM element has the correct styling but let ComfyUI position it
        if (widget.element) {
            widget.element.style.height = interfaceHeight + 'px';
            widget.element.style.minHeight = interfaceHeight + 'px';
            widget.element.style.display = 'block';
            widget.element.style.position = 'relative'; // Let ComfyUI handle positioning
            widget.element.style.boxSizing = 'border-box';
            widget.element.style.width = '100%';
            widget.element.style.maxWidth = 'none';
            widget.element.style.minWidth = '100%';
            widget.element.style.alignSelf = 'stretch';
            widget.element.style.margin = '0';
            widget.element.style.padding = '0';
            widget.element.style.pointerEvents = 'auto';

            if (widget.name === 'audio_analyzer_button_row') {
                widget.element.style.height = interfaceHeight + 'px';
                widget.element.style.minHeight = interfaceHeight + 'px';
                widget.element.style.background = 'transparent';
                widget.element.style.display = 'flex';
                widget.element.style.justifyContent = 'center';
                widget.element.style.alignItems = 'center';
            }
        }
    }

    insertSpacerWidget() {
        if (!this.core.node.widgets) return null;
        const existingSpacerIndex = this.core.node.widgets.findIndex(w => w.name === 'audio_analyzer_spacer');
        if (existingSpacerIndex >= 0) {
            this.core.node.widgets.splice(existingSpacerIndex, 1);
        }

        const spacerHeight = this.bottomSpacerHeight;
        const spacerWidget = {
            type: 'audio_analyzer_bottom_spacer',
            name: 'audio_analyzer_spacer',
            value: '',
            serialize: false,
            computedHeight: spacerHeight,
            computeSize(width) {
                return [width || 760, spacerHeight];
            },
            getHeight() {
                return spacerHeight;
            },
            draw() {
                // Reserve a real ComfyUI layout row below the DOM panel.
            },
            mouse() {
                return false;
            }
        };

        this.core.node.widgets.push(spacerWidget);
        return spacerWidget;
    }

    positionInterfaceOverSpacer() {
        // Kept as a no-op for compatibility with older call sites.
    }

    getWidgetHeight(widget) {
        if (widget.name === 'audio_analyzer_interface') {
            return 0;
        }

        if (typeof widget.getHeight === 'function') {
            const height = Number(widget.getHeight());
            if (Number.isFinite(height) && height > 0) {
                return height;
            }
        }

        if (typeof widget.computeSize === 'function') {
            const size = widget.computeSize(this.core.node?.size?.[0] || 760);
            const height = Number(size?.[1]);
            if (Number.isFinite(height) && height > 0) {
                return height;
            }
        }

        const explicitHeight = Number(widget.computedHeight || widget.height);
        if (Number.isFinite(explicitHeight) && explicitHeight > 0) {
            return explicitHeight;
        }

        if (widget.type === 'string' && widget.options && widget.options.multiline) {
            const lines = Math.max(1, (widget.value || '').split('\n').length);
            return Math.max(200, lines * 20 + 20);
        }

        return 30;
    }

    findNodeElement() {
        // Try multiple methods to find the node's DOM element

        // Method 1: Look for node by ID
        if (this.core.node.id) {
            let nodeElement = document.querySelector(`[data-id="${this.core.node.id}"]`);
            if (nodeElement) return nodeElement;
        }

        // Method 2: Look for the node through LiteGraph canvas
        try {
            const canvas = this.core.node.graph?.canvas;
            if (canvas && canvas.canvas) {
                // Find the node element within the canvas container
                const canvasContainer = canvas.canvas.parentElement;
                if (canvasContainer) {
                    // Look for elements that might be our node
                    const nodeElements = canvasContainer.querySelectorAll('.litegraph-node, .node, [class*="node"]');
                    for (let element of nodeElements) {
                        // This is a rough check - in a real implementation you'd need more specific identification
                        if (element.textContent && element.textContent.includes('Audio Wave Analyzer')) {
                            return element;
                        }
                    }
                }
            }
        } catch (error) {
            console.log('Failed to find node through canvas:', error);
        }

        // Method 3: Fallback to a container that can handle absolute positioning
        return document.body;
    }

    findInsertPosition() {
        // Find position after energy_sensitivity to insert our spacer
        if (!this.core.node.widgets) return 0;

        const widgets = this.core.node.widgets;

        // Always insert at the very end
        // console.log(`🎵 Inserting spacer at end position ${widgets.length}`);  // Debug: spacer position
        return widgets.length;
    }

    setupMultilineWidgetWatchers() {
        // Watch for changes in multiline widgets that might affect node height
        if (!this.core.node.widgets) return;

        this.core.node.widgets.forEach(widget => {
            if (widget.type === 'string' && widget.options && widget.options.multiline) {
                // Store original callback
                const originalCallback = widget.callback;

                // Wrap callback to recalculate height on changes
                widget.callback = (value) => {
                    if (originalCallback) {
                        originalCallback.call(widget, value);
                    }

                    // Recalculate and update node height after a short delay
                    setTimeout(() => {
                        this.recalculateNodeHeight();
                    }, 50);
                };
            }
        });
    }

    recalculateNodeHeight() {
        // Recalculate node height based on current widget content
        if (!this.core.node || !this.core.layout) return;

        const computedHeight = this.core.layout.getTargetNodeHeight();

        // Update node size if height changed significantly
        if (this.core.node.size && Math.abs(this.core.node.size[1] - computedHeight) > 1) {
            this.core.layout?.setNodeSize?.(
                this.core.node.size[0],
                computedHeight
            );
        }
    }

    calculateWidgetHeights(widgets) {
        // Calculate total height needed for widgets
        let otherWidgetsHeight = 0;

        widgets.forEach(widget => {
            otherWidgetsHeight += this.getWidgetHeight(widget);
        });

        return otherWidgetsHeight;
    }

    ensureUIVisible() {
        // Ensure UI is visible in the spacer area
        if (!this.core.container) {
            console.log('🎵 UI container not found for visibility check');
            return;
        }

        // Make sure container is visible and positioned correctly
        const container = this.core.container;

        // Ensure container has proper styling for visibility
        container.style.display = 'block';
        container.style.visibility = 'visible';
        container.style.opacity = '1';

        // Add to node's DOM if not already present
        if (!container.parentElement) {
            // Find the node's DOM element
            const nodeElement = document.querySelector(`[data-id="${this.core.node.id}"]`) ||
                              this.core.node.graph?.canvas?.canvas?.parentElement;

            if (nodeElement) {
                nodeElement.appendChild(container);
                console.log('🌊 Audio Wave Analyzer: Added container to node DOM');
            } else {
                // Fallback to body
                document.body.appendChild(container);
                console.log('🌊 Audio Wave Analyzer: Added container to body as fallback');
            }
        }
    }
}
