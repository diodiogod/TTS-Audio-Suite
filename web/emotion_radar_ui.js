/**
 * Emotion Radar Chart UI Module
 * Handles interface creation and DOM management
 */
export class EmotionRadarUI {
    constructor(core) {
        this.core = core;
        this.container = null;
        this.canvas = null;
        this.controlsContainer = null;
        this.messageDisplay = null;
        this.interfaceHeight = 310; // Compact height
    }

    createInterface() {
        // Remove existing interface if it exists
        this.removeExistingInterface();

        // Create main container with elegant styling
        this.container = this.createMainContainer();

        // Create canvas
        this.canvas = this.createCanvas();

        // Create controls
        this.controlsContainer = this.createControls();

        // Assemble interface
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controlsContainer);

        // Add to node
        this.addContainerToNode();

        // Setup canvas references
        this.core.canvas = this.canvas;
        this.core.ctx = this.canvas.getContext('2d');

        // Setup initial canvas size
        this.setupCanvasSize();
    }

    removeExistingInterface() {
        const existingInterface = this.core.node.widgets?.find(w => w.name === 'emotion_radar_interface');
        if (existingInterface && existingInterface.element) {
            const existingContainer = existingInterface.element;
            if (existingContainer.parentNode) {
                existingContainer.parentNode.removeChild(existingContainer);
            }
        }
    }

    createMainContainer() {
        const container = document.createElement('div');
        container.style.cssText = `
            width: 100%;
            height: ${this.interfaceHeight}px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            border: 1px solid #444;
            border-radius: 8px;
            padding: 8px;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
            position: relative;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        `;
        return container;
    }

    createCanvas() {
        const canvas = document.createElement('canvas');
        canvas.style.cssText = `
            width: 270px;
            height: 270px;
            border-radius: 6px;
            background: radial-gradient(circle at center, #2a2a2a 0%, #1e1e1e 70%, #1a1a1a 100%);
            cursor: default;
            display: block;
            margin: 0 auto 6px auto;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        `;
        return canvas;
    }

    createControls() {
        const controls = document.createElement('div');
        controls.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            padding: 4px;
            height: 24px;
        `;

        // Reset button
        const resetBtn = this.createButton('Reset', '#ff6b6b', () => {
            this.core.resetAllEmotions();
        });

        // Random button
        const randomBtn = this.createButton('Random', '#4ecdc4', () => {
            this.core.randomizeEmotions();
        });

        // Export button
        const exportBtn = this.createButton('Export', '#45b7d1', () => {
            this.core.exportEmotionConfig();
        });

        // Message display
        this.messageDisplay = document.createElement('div');
        this.messageDisplay.style.cssText = `
            color: #ccc;
            font-size: 10px;
            text-align: center;
            margin-left: auto;
            padding: 2px 6px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 3px;
            min-width: 100px;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        this.messageDisplay.textContent = 'Click and drag to set emotions';

        controls.appendChild(resetBtn);
        controls.appendChild(randomBtn);
        controls.appendChild(exportBtn);
        controls.appendChild(this.messageDisplay);

        return controls;
    }

    createButton(text, color, onClick) {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.cssText = `
            background: ${color};
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        `;

        button.addEventListener('mouseenter', () => {
            button.style.transform = 'translateY(-1px)';
            button.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.4)';
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = 'translateY(0)';
            button.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.3)';
        });

        button.addEventListener('click', onClick);
        return button;
    }

    addContainerToNode() {
        // Use the exact Audio Wave Analyzer pattern
        // Create a simple spacer widget that reserves space
        const spacerWidget = this.createSpacerWidget();

        // Add widget to node
        if (!this.core.node.widgets) {
            this.core.node.widgets = [];
        }

        // Insert spacer at the end
        this.core.node.widgets.push(spacerWidget);

        // Add the DOM container to the document
        document.body.appendChild(this.container);

        // Position it over the spacer widget (this happens during widget draw)
        this.positionInterfaceOverSpacer();

        console.log('🎭 Emotion Radar Chart: Interface created with Audio Wave Analyzer pattern');
    }

    createSpacerWidget() {
        const interfaceHeight = this.interfaceHeight;

        return {
            type: 'blank_spacer',
            name: 'emotion_radar_spacer',
            value: '',
            height: interfaceHeight,
            computedHeight: interfaceHeight,
            serialize: false,

            // Reserve space in layout calculations
            computeSize: function(width) {
                return [width || 300, interfaceHeight];
            },

            getHeight: function() {
                return interfaceHeight;
            },

            // Draw a simple placeholder
            draw: (ctx, node, widget_width, y, widget_height) => {
                // Draw subtle background to show reserved space
                ctx.fillStyle = '#2a2a2a';
                ctx.fillRect(0, y, widget_width, Math.min(widget_height, interfaceHeight));

                // Store Y position for interface positioning
                this.renderY = y;
            },

            // Non-interactive
            mouse: function(event, pos, node) {
                return false;
            }
        };
    }

    positionInterfaceOverSpacer() {
        // Position our DOM interface to render over the spacer widget
        // This is the exact same method from AudioAnalyzerWidgets
        if (!this.container || !this.core.node.widgets) {
            console.log('🎭 Container not found for positioning');
            return;
        }

        const spacerWidget = this.core.node.widgets.find(w => w.name === 'emotion_radar_spacer');
        if (!spacerWidget) {
            console.log('🎭 Spacer widget not found for positioning');
            return;
        }

        // Wait for the spacer to be rendered and get its position
        const checkPosition = () => {
            if (spacerWidget.renderY !== undefined) {
                // Find the node's DOM element to position relative to it
                const nodeElement = this.findNodeElement();
                if (nodeElement) {
                    // Position relative to the node element
                    this.container.style.position = 'absolute';
                    this.container.style.top = `${spacerWidget.renderY}px`;
                    this.container.style.left = '10px'; // Small margin from node edge
                    this.container.style.width = 'calc(100% - 20px)'; // Account for margins
                    this.container.style.zIndex = '10';
                    this.container.style.pointerEvents = 'auto';

                    // Append to node element instead of body
                    if (this.container.parentElement !== nodeElement) {
                        nodeElement.appendChild(this.container);
                    }

                    console.log(`🎭 Emotion Radar Chart: Positioned interface over spacer at Y=${spacerWidget.renderY}px within node`);
                } else {
                    console.log('🎭 Could not find node element for positioning');
                }
            } else {
                // Retry positioning after a short delay
                setTimeout(checkPosition, 100);
            }
        };

        // Start checking after a short delay to allow rendering
        setTimeout(checkPosition, 200);
    }

    setupCanvasSize() {
        if (!this.canvas) return;

        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.core.ctx.scale(devicePixelRatio, devicePixelRatio);

        // Update center coordinates
        this.core.centerX = rect.width / 2;
        this.core.centerY = rect.height / 2;
    }

    setupCanvasResize() {
        if (!this.canvas) return;

        const resizeObserver = new ResizeObserver(() => {
            this.core.resizeCanvas();
        });

        resizeObserver.observe(this.canvas);
    }

    showMessage(message) {
        if (!this.messageDisplay) return;

        this.messageDisplay.textContent = message;
        this.messageDisplay.style.opacity = '1';

        // Clear previous timeout
        if (this.messageTimeout) {
            clearTimeout(this.messageTimeout);
        }

        // Fade out after 2 seconds
        this.messageTimeout = setTimeout(() => {
            this.messageDisplay.style.opacity = '0';
        }, 2000);
    }

    // Utility method to position interface correctly within ComfyUI
    positionInterface() {
        if (!this.container || !this.core.node) return;

        // Try to position relative to the node
        const nodeElement = this.findNodeElement();
        if (nodeElement) {
            // Position within the node's widget area
            this.container.style.position = 'relative';
            this.container.style.zIndex = '10';
        } else {
            // Fallback positioning
            this.container.style.position = 'absolute';
            this.container.style.zIndex = '1000';
        }
    }

    findNodeElement() {
        // Try multiple methods to find the node's DOM element
        // This is the exact same method from AudioAnalyzerWidgets

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
                        if (element.textContent && element.textContent.includes('IndexTTS-2 Emotion')) {
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
}