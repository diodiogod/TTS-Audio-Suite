/**
 * Audio Wave Analyzer Widget Management Module
 * Handles widget creation, positioning, and spacer management
 */
export class AudioAnalyzerWidgets {
    constructor(core) {
        this.core = core;
        this.interfaceHeight = 420;
    }
    
    setupWidgetHeight(widget) {
        // Ensure the widget properly reserves its height in ComfyUI's layout system
        const interfaceHeight = this.interfaceHeight;
        
        // Set the widget to properly reserve space in the layout
        widget.height = interfaceHeight;
        widget.computedHeight = interfaceHeight;
        
        // Override ComfyUI's height calculation methods
        widget.computeSize = function(width) {
            return [width || 780, interfaceHeight]; // Reserve full height in layout
        };
        
        widget.getHeight = function() {
            return interfaceHeight;
        };
        
        // Simple draw method that doesn't interfere with positioning
        widget.draw = function(ctx, node, widget_width, y, widget_height) {
            // Draw a placeholder rectangle to show where the widget should be
            ctx.fillStyle = '#2a2a2a';
            ctx.fillRect(0, y, widget_width, Math.min(widget_height, interfaceHeight));
            
            // Don't try to position the DOM element here - let ComfyUI handle it
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
            widget.element.style.maxWidth = '780px';
        }
        
        console.log(`🌊 Audio Wave Analyzer: Setup widget with reserved height: ${interfaceHeight}px`);
    }
    
    createBlankSpacerWidget() {
        // Create a simple spacer widget that just reserves space
        const interfaceHeight = this.interfaceHeight;
        
        const spacerWidget = {
            type: 'blank_spacer',
            name: 'audio_analyzer_spacer',
            value: '',
            height: interfaceHeight,
            computedHeight: interfaceHeight,
            serialize: false,
            
            // Reserve space in layout calculations
            computeSize: function(width) {
                return [width || 780, interfaceHeight];
            },
            
            getHeight: function() {
                return interfaceHeight;
            },
            
            // Draw a simple placeholder
            draw: function(ctx, node, widget_width, y, widget_height) {
                // Draw subtle background to show reserved space
                ctx.fillStyle = '#2a2a2a';
                ctx.fillRect(0, y, widget_width, Math.min(widget_height, interfaceHeight));
                
                // Add label for debugging
                ctx.fillStyle = '#666666';
                ctx.font = '12px Arial';
                ctx.fillText('Interface will replace this spacer', 10, y + 20);
            },
            
            // Non-interactive
            mouse: function(event, pos, node) {
                return false;
            }
        };
        
        return spacerWidget;
    }
    
    insertSpacerWidget() {
        // Insert the blank spacer widget at the right position
        if (!this.core.node.widgets) return null;
        
        const spacerWidget = this.createBlankSpacerWidget();
        const insertPosition = this.findInsertPosition();
        
        // Insert spacer widget
        this.core.node.widgets.splice(insertPosition, 0, spacerWidget);
        
        // console.log(`🌊 Audio Wave Analyzer: Inserted blank spacer widget at position ${insertPosition}`);  // Debug: spacer insertion
        return spacerWidget;
    }
    
    positionInterfaceOverSpacer() {
        // Position our DOM interface to render over the spacer widget
        if (!this.core.container || !this.core.node.widgets) {
            console.log('🎵 Container not found for positioning');
            return;
        }
        
        const spacerWidget = this.core.node.widgets.find(w => w.name === 'audio_analyzer_spacer');
        if (!spacerWidget) {
            console.log('🎵 Spacer widget not found for positioning');
            return;
        }
        
        // Wait for the spacer to be rendered and get its position
        const checkPosition = () => {
            if (spacerWidget.renderY !== undefined) {
                // Find the node's DOM element to position relative to it
                const nodeElement = this.findNodeElement();
                if (nodeElement) {
                    // Position relative to the node element
                    this.core.container.style.position = 'absolute';
                    this.core.container.style.top = `${spacerWidget.renderY}px`;
                    this.core.container.style.left = '10px'; // Small margin from node edge
                    this.core.container.style.width = 'calc(100% - 20px)'; // Account for margins
                    this.core.container.style.zIndex = '10';
                    this.core.container.style.pointerEvents = 'auto';
                    
                    // Append to node element instead of body
                    if (this.core.container.parentElement !== nodeElement) {
                        nodeElement.appendChild(this.core.container);
                    }
                    
                    console.log(`🌊 Audio Wave Analyzer: Positioned interface over spacer at Y=${spacerWidget.renderY}px within node`);
                } else {
                    console.log('🎵 Could not find node element for positioning');
                }
            } else {
                // Retry positioning after a short delay
                setTimeout(checkPosition, 100);
            }
        };
        
        // Start checking after a short delay to allow rendering
        setTimeout(checkPosition, 200);
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
        if (!this.core.node) return;
        
        const interfaceHeight = this.interfaceHeight;
        const widgets = this.core.node.widgets || [];
        let otherWidgetsHeight = 0;
        
        widgets.forEach(widget => {
            if (widget.name === 'audio_analyzer_interface') {
                return; // Skip our interface widget
            }
            
            // Check if it's a multiline widget
            if (widget.type === 'string' && widget.options && widget.options.multiline) {
                // Multiline widgets are taller
                const lines = Math.max(1, (widget.value || '').split('\n').length);
                otherWidgetsHeight += Math.max(60, lines * 20 + 20); // Minimum 60px, or based on content
            } else {
                // Regular widgets
                otherWidgetsHeight += 30;
            }
        });
        
        const nodeExtraHeight = 100; // Space for node title, margins, etc.
        const newHeight = interfaceHeight + otherWidgetsHeight + nodeExtraHeight;
        
        // Update node size if height changed significantly
        if (this.core.node.size && Math.abs(this.core.node.size[1] - newHeight) > 10) {
            this.core.node.size[1] = newHeight;
            
            if (this.core.node.onResize) {
                this.core.node.onResize(this.core.node.size);
            }
            
            console.log(`🌊 Audio Wave Analyzer: Recalculated node height to ${newHeight}px`);
        }
    }
    
    calculateWidgetHeights(widgets) {
        // Calculate total height needed for widgets
        let otherWidgetsHeight = 0;
        
        widgets.forEach(widget => {
            if (widget.name === 'audio_analyzer_interface') {
                return; // Skip our interface widget
            }
            
            // Check if it's a multiline widget
            if (widget.type === 'string' && widget.options && widget.options.multiline) {
                // Multiline widgets are taller
                const lines = Math.max(1, (widget.value || '').split('\n').length);
                otherWidgetsHeight += Math.max(60, lines * 20 + 20); // Minimum 60px, or based on content
            } else {
                // Regular widgets
                otherWidgetsHeight += 30;
            }
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