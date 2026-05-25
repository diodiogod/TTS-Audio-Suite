/**
 * Audio Wave Analyzer Layout Management Module
 * Handles node sizing, positioning, and resize behavior
 */
export class AudioAnalyzerLayout {
    constructor(core) {
        this.core = core;
        this.buttonRowHeight = 44;
        this.interfaceHeight = 420;
        this.nodeBottomPadding = 48;
        this.minInterfaceWidth = 760;
    }

    getTargetNodeHeight() {
        let requiredHeight = this.interfaceHeight + this.buttonRowHeight + 120;
        if (typeof this.core.node?.computeSize === 'function') {
            requiredHeight = Math.max(requiredHeight, Number(this.core.node.computeSize()?.[1]) || 0);
        } else {
            const widgets = this.core.node?.widgets || [];
            requiredHeight = this.core.widgets.calculateWidgetHeights(widgets) + 80;
        }
        return Math.ceil(requiredHeight + this.nodeBottomPadding);
    }

    resizeNodeForInterface() {
        // Ensure the ComfyUI node is properly sized to contain our interface
        if (!this.core.node) return;

        try {
            const requiredWidth = this.minInterfaceWidth;
            const requiredHeight = this.getTargetNodeHeight();
            this.setNodeSize(
                Math.max(this.core.node.size?.[0] || 0, requiredWidth),
                requiredHeight
            );

        } catch (error) {
            console.error('Failed to resize node for interface:', error);
        }
    }

    setNodeSize(width, height) {
        if (!this.core.node) return;

        const nextSize = [
            Math.max(Number(width) || 0, this.minInterfaceWidth),
            Math.ceil(Number(height) || 0),
        ];

        if (typeof this.core.node.setSize === 'function') {
            this.core.node.setSize(nextSize);
        } else {
            this.core.node.size = nextSize;
            if (typeof this.core.node.onResize === 'function') {
                this.core.node.onResize(nextSize);
            }
        }

        if (this.core.node.graph?.setDirtyCanvas) {
            this.core.node.graph.setDirtyCanvas(true, true);
        }
    }

    updateNodeLayout() {
        // Force ComfyUI to update the node layout after adding our interface
        if (!this.core.node) return;

        try {
            // Trigger a layout update
            if (this.core.node.graph && this.core.node.graph.setDirtyCanvas) {
                this.core.node.graph.setDirtyCanvas(true, true);
            }

            // Schedule a delayed resize to ensure everything is rendered
            setTimeout(() => {
                if (this.core.node.setDirtyCanvas) {
                    this.core.node.setDirtyCanvas(true);
                }
            }, 100);

        } catch (error) {
            console.error('Failed to update node layout:', error);
        }
    }

    setupNodeResizeHandling() {
        // Setup resize handling
        if (!this.core.node) return;

        this.core.node.resizable = true;

        // Store original resize method
        const originalOnResize = this.core.node.onResize;

        // Custom resize handler that keeps the widget edge-to-edge and rerenders
        // the canvas at the current graph zoom/DPR.
        this.core.node.onResize = (size) => {
            // Prevent infinite resize loops
            if (this.resizing) return;
            this.resizing = true;

            const constrainedSize = [
                Math.max(size[0], this.minInterfaceWidth),
                Math.max(size[1], this.getTargetNodeHeight())
            ];

            if (originalOnResize) {
                originalOnResize.call(this.core.node, constrainedSize);
            }

            // Ensure our canvas resizes with the interface
            if (this.core.canvas) {
                this.core.resizeCanvas();
            }

            setTimeout(() => { this.resizing = false; }, 10);
        };

        // Store reference to interface for access in node methods
        this.core.node.audioAnalyzerInterface = this.core;

        // Hook into node removal for cleanup
        const originalOnRemoved = this.core.node.onRemoved;
        this.core.node.onRemoved = () => {
            if (originalOnRemoved) {
                originalOnRemoved.call(this.core.node);
            }
            this.destroy();
        };

        // Watch for changes in multiline widgets to recalculate height
        this.core.widgets.setupMultilineWidgetWatchers();
    }

    destroy() {
        // Cleanup when the interface is destroyed
        if (this.positionUpdateInterval) {
            clearInterval(this.positionUpdateInterval);
            this.positionUpdateInterval = null;
        }

        if (this.core.widget && this.core.widget.element && this.core.widget.element.parentElement) {
            this.core.widget.element.parentElement.removeChild(this.core.widget.element);
        }

        console.log('🌊 Audio Wave Analyzer Layout destroyed and cleaned up');
    }

    createMainContainer() {
        // Create main container
        const container = document.createElement('div');
        container.className = 'audio-analyzer-container';
        container.style.cssText = `
            width: 100%;
            height: ${this.interfaceHeight}px;
            min-height: ${this.interfaceHeight}px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            box-sizing: border-box;
            overflow: hidden;
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 12px;
            color: #ffffff;
        `;

        return container;
    }

    createCanvas() {
        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.className = 'audio-analyzer-canvas';
        canvas.style.cssText = `
            width: 100%;
            height: 300px;
            background: #1a1a1a;
            display: block;
            cursor: crosshair;
            image-rendering: auto;
        `;

        return canvas;
    }

    createControlsContainer() {
        // Create controls container
        const controls = document.createElement('div');
        controls.className = 'audio-analyzer-controls';
        controls.style.cssText = `
            height: 80px;
            background: #2a2a2a;
            border-top: 1px solid #333;
            padding: 8px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;

        return controls;
    }

    createAnalyzeButtonRow() {
        const row = document.createElement('div');
        row.className = 'audio-analyzer-button-row';
        row.style.cssText = `
            width: 100%;
            height: ${this.buttonRowHeight}px;
            min-height: ${this.buttonRowHeight}px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            pointer-events: auto;
        `;

        if (this.core.analyzeButton) {
            row.appendChild(this.core.analyzeButton);
        }

        return row;
    }

    addContainerToNode(container) {
        // Use ComfyUI's DOM widget as the actual layout element. The old
        // spacer/absolute-position approach drifted with widget heights and zoom.
        try {
            // Ensure the node is properly sized to contain our interface
            this.resizeNodeForInterface();

            if (typeof this.core.node.addDOMWidget === 'function') {
                const buttonRow = this.createAnalyzeButtonRow();
                const buttonWidget = this.core.node.addDOMWidget('audio_analyzer_button_row', 'div', buttonRow, {
                    serialize: false,
                    hideOnZoom: false,
                    height: this.buttonRowHeight
                });

                this.core.buttonWidget = buttonWidget;
                this.core.widgets.setupWidgetHeight(buttonWidget, this.buttonRowHeight);

                const widget = this.core.node.addDOMWidget('audio_analyzer_interface', 'div', container, {
                    serialize: false,
                    hideOnZoom: false,
                    height: this.interfaceHeight
                });

                this.core.widget = widget;
                this.core.widgets.setupWidgetHeight(widget);
                this.core.spacerWidget = this.core.widgets.insertSpacerWidget();
            } else {
                console.log('🌊 Audio Wave Analyzer: addDOMWidget not available, using fallback');
                return false;
            }

            this.resizeNodeForInterface();

            // Ensure node layout is updated
            this.updateNodeLayout();

            // Setup simplified node resize handling
            this.setupNodeResizeHandling();

            const finalizeLayout = () => {
                this.resizeNodeForInterface();
                this.core.widgets.recalculateNodeHeight();
                this.core.resizeCanvas();
            };
            requestAnimationFrame(finalizeLayout);
            setTimeout(finalizeLayout, 100);

            return true;

        } catch (error) {
            console.error('Failed to add container as widget:', error);
            return false;
        }
    }

    positionInterfaceOverSpacer() {}
}
