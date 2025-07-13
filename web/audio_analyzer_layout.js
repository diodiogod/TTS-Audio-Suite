/**
 * Audio Wave Analyzer Layout Management Module
 * Handles node sizing, positioning, and resize behavior
 */
export class AudioAnalyzerLayout {
    constructor(core) {
        this.core = core;
        this.interfaceHeight = 420;
        this.interfaceWidth = 800;
    }
    
    resizeNodeForInterface() {
        // Ensure the ComfyUI node is properly sized to contain our interface
        if (!this.core.node) return;
        
        try {
            const interfaceHeight = this.interfaceHeight;
            const interfaceWidth = this.interfaceWidth;
            
            // Calculate widget heights more accurately
            const widgets = this.core.node.widgets || [];
            const otherWidgetsHeight = this.core.widgets.calculateWidgetHeights(widgets);
            
            const nodeExtraHeight = 100; // Space for node title, margins, etc. (increased for multiline widgets)
            
            const requiredWidth = Math.max(interfaceWidth, 850);
            const requiredHeight = interfaceHeight + otherWidgetsHeight + nodeExtraHeight;
            
            if (this.core.node.size) {
                this.core.node.size[0] = Math.max(this.core.node.size[0], requiredWidth);
                this.core.node.size[1] = Math.max(this.core.node.size[1], requiredHeight);
            } else {
                this.core.node.size = [requiredWidth, requiredHeight];
            }
            
            // Force node to update its layout
            if (this.core.node.onResize) {
                this.core.node.onResize(this.core.node.size);
            }
            
            // console.log(`🌊 Audio Wave Analyzer: Resized node to ${this.core.node.size[0]}x${this.core.node.size[1]} for ${widgets.length} widgets (${otherWidgetsHeight}px for other widgets)`);  // Debug: node resize
            
        } catch (error) {
            console.error('Failed to resize node for interface:', error);
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
        // Setup horizontal-only resize handling
        if (!this.core.node) return;
        
        // Calculate fixed height based on interface + widgets
        const interfaceHeight = this.interfaceHeight;
        const widgets = this.core.node.widgets || [];
        const otherWidgetsHeight = this.core.widgets.calculateWidgetHeights(widgets);
        
        const nodeExtraHeight = 100; // Space for node title, margins, etc.
        const fixedHeight = 950; // Fixed test height
        
        // Set node properties to control resizing
        this.core.node.resizable = true; // Allow resizing
        this.core.node.horizontal_resize_only = true; // Custom property to indicate horizontal-only resize
        
        // Store original resize method
        const originalOnResize = this.core.node.onResize;
        
        // Custom resize handler that constrains vertical resizing
        this.core.node.onResize = (size) => {
            // Prevent infinite resize loops
            if (this.resizing) return;
            this.resizing = true;
            
            // Only constrain width, don't force height changes
            const constrainedSize = [
                Math.max(size[0], 850), // Minimum width
                size[1] // Allow ComfyUI to control height
            ];
            
            // Only update if width changed significantly
            if (Math.abs(this.core.node.size[0] - constrainedSize[0]) > 10) {
                this.core.node.size = constrainedSize;
                
                // Call original resize if it exists
                if (originalOnResize) {
                    originalOnResize.call(this.core.node, constrainedSize);
                }
                
                console.log(`🌊 Audio Wave Analyzer: Node width constrained to ${constrainedSize[0]}px`);
            }
            
            // Ensure our canvas resizes with the interface
            if (this.core.canvas) {
                this.core.resizeCanvas();

            }
            
            setTimeout(() => { this.resizing = false; }, 10);
        };
        
        // Override mouse handling for resize to prevent vertical resize
        const originalOnMouseMove = this.core.node.onMouseMove;
        this.core.node.onMouseMove = function(e, pos, node_graph_pos) {
            // Check if we're in a resize operation
            if (this.flags && this.flags.resizing) {
                // Only allow horizontal resizing
                if (this.size) {
                    this.size[1] = fixedHeight; // Keep height fixed
                }
            }
            
            // Call original mouse move handler
            if (originalOnMouseMove) {
                return originalOnMouseMove.call(this, e, pos, node_graph_pos);
            }
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
        
        // Set initial size to fixed height
        if (this.core.node.size) {
            this.core.node.size[1] = fixedHeight;
        }
        
        // console.log(`🌊 Audio Wave Analyzer: Setup horizontal-only resize handling (fixed height: ${fixedHeight}px)`);  // Debug: resize setup
        
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
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
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
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;
        
        return controls;
    }
    
    addContainerToNode(container) {
        // Use ComfyUI's widget system but position over spacer
        try {
            // Ensure the node is properly sized to contain our interface
            this.resizeNodeForInterface();
            
            if (typeof this.core.node.addDOMWidget === 'function') {
                // Create a zero-height widget that ComfyUI positions normally
                const widget = this.core.node.addDOMWidget('audio_analyzer_interface', 'div', container, {
                    serialize: false,
                    hideOnZoom: false,
                    height: 0 // Zero height so it doesn't take space
                });
                
                // Make the container position absolutely within the node
                container.style.position = 'absolute';
                container.style.height = `${this.interfaceHeight}px`;
                container.style.width = '100%';
                container.style.zIndex = '10';
                container.style.pointerEvents = 'auto';
                
                this.core.widget = widget;
                
                // console.log('🌊 Audio Wave Analyzer: Interface widget added with absolute positioning');  // Debug: widget positioning
                
                // Position the interface over the spacer after a short delay to ensure spacer exists
                setTimeout(() => {
                    this.positionInterfaceOverSpacer();
                }, 100);
                
            } else {
                console.log('🌊 Audio Wave Analyzer: addDOMWidget not available, using fallback');
                return false;
            }
            
            // Ensure node layout is updated
            this.updateNodeLayout();
            
            // Setup simplified node resize handling
            this.setupNodeResizeHandling();
            
            return true;
            
        } catch (error) {
            console.error('Failed to add container as widget:', error);
            return false;
        }
    }
    
    positionInterfaceOverSpacer() {
        // Calculate where the spacer is within the node and position interface there
        if (!this.core.node.widgets || !this.core.widget) return;
        
        const spacerIndex = this.core.node.widgets.findIndex(w => w.name === 'audio_analyzer_spacer');
        if (spacerIndex === -1) return;
        
        // Calculate the Y position more accurately
        let yPosition = 40; // Start with just node title space
        
        for (let i = 0; i < spacerIndex; i++) {
            const widget = this.core.node.widgets[i];
            if (widget.name !== 'audio_analyzer_interface') {
                // Use smaller, more accurate widget heights
                if (widget.type === 'string' && widget.options && widget.options.multiline) {
                    const lines = Math.max(1, (widget.value || '').split('\n').length);
                    yPosition += Math.max(40, lines * 18 + 10); // Reduced height
                } else {
                    yPosition += 25; // Reduced standard widget height
                }
            }
        }
        
        // Position the interface container - move it WAY up
        const adjustedPosition = yPosition - 178; // Move up offset px. was 250
        this.core.widget.element.style.top = `${adjustedPosition}px`;
        
        // console.log(`🌊 Audio Wave Analyzer: Positioned interface at node Y=${yPosition}px over spacer`);  // Debug: interface positioning
    }
}