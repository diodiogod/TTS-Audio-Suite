/**
 * Audio Analyzer UI Module
 * Handles all DOM creation and UI management
 */
export class AudioAnalyzerUI {
    constructor(core) {
        this.core = core;
        this.container = null;
        this.canvas = null;
        this.playButton = null;
        this.stopButton = null;
        this.timeDisplay = null;
        this.selectionDisplay = null;
        this.statusDisplay = null;
        this.messageDisplay = null;
        this.controls = null;
        this.analysisControls = null;
        this.regionControls = null;
        this.exportControls = null;
        this.zoomControls = null;
    }
    
    createInterface() {
        // Create Audio Analyzer UI interface
        
        // Remove existing interface
        const existingInterface = this.core.node.widgets?.find(w => w.name === 'audio_analyzer_interface');
        if (existingInterface) {
            const existingContainer = existingInterface.element;
            if (existingContainer && existingContainer.parentNode) {
                existingContainer.parentNode.removeChild(existingContainer);
            }
        }
        
        // Create main container
        this.container = document.createElement('div');
        this.container.className = 'audio-analyzer-container';
        this.container.style.cssText = `
            width: 100%;
            height: 420px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 12px;
            color: #ffffff;
        `;
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'audio-analyzer-canvas';
        this.canvas.style.cssText = `
            width: 100%;
            height: 300px;
            background: #1a1a1a;
            display: block;
            cursor: crosshair;
        `;
        
        // Get canvas context
        this.core.canvas = this.canvas;
        this.core.ctx = this.canvas.getContext('2d');
        
        // Create controls container
        this.controls = document.createElement('div');
        this.controls.className = 'audio-analyzer-controls';
        this.controls.style.cssText = `
            height: 80px;
            background: #2a2a2a;
            border-top: 1px solid #333;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;
        
        // Create playback controls
        this.createPlaybackControls();
        
        // Create main action controls (consolidated)
        this.createMainControls();
        
        // Create zoom controls
        this.createZoomControls();
        
        // Create status displays
        this.createStatusDisplays();
        
        // Assemble interface
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controls);
        
        // Add container to node using ComfyUI's DOM widget system
        try {
            // Ensure the node is properly sized to contain our interface
            this.resizeNodeForInterface();
            
            if (typeof this.core.node.addDOMWidget === 'function') {
                const widget = this.core.node.addDOMWidget('audio_analyzer_interface', 'div', this.container, {
                    serialize: false,
                    hideOnZoom: false,
                    height: 420 // Explicit height for our interface
                });
                
                // Ensure the widget properly reports its height to ComfyUI layout
                this.setupWidgetHeight(widget);
                this.widget = widget;
                
                // Move widget to earlier position to avoid overflow from multiline widgets
                this.repositionWidget(widget);
                
            } else {
                // Alternative method: create custom widget
                const widget = {
                    type: 'div',
                    name: 'audio_analyzer_interface',
                    element: this.container,
                    value: null,
                    serialize: false,
                    hideOnZoom: false,
                    height: 420,
                    draw: function(ctx, node, widget_width, widget_height) {
                        // This will be called for custom drawing
                    }
                };
                
                // Ensure the widget properly reports its height to ComfyUI layout
                this.setupWidgetHeight(widget);
                
                this.core.node.widgets = this.core.node.widgets || [];
                
                // Insert at position to avoid being pushed down by multiline widgets
                const insertPosition = this.findInsertPosition();
                this.core.node.widgets.splice(insertPosition, 0, widget);
                this.widget = widget;
            }
            
            // Ensure node layout is updated
            this.updateNodeLayout();
            
            // Hook into node resize events to keep interface positioned correctly
            this.setupNodeResizeHandling();
            
            // Initial positioning
            setTimeout(() => this.repositionInterface(), 100);
            
        } catch (error) {
            console.error('Failed to add DOM widget:', error);
        }
        
        // Setup initial canvas size
        this.setupCanvasSize();
        
        // Force UI to appear by adding to DOM
        this.ensureUIVisible();
    }
    
    createPlaybackControls() {
        const playbackContainer = document.createElement('div');
        playbackContainer.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Play button
        this.playButton = document.createElement('button');
        this.playButton.textContent = '▶️ Play';
        this.playButton.style.cssText = `
            padding: 4px 8px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.playButton.onclick = () => this.core.togglePlayback();
        
        // Stop button
        this.stopButton = document.createElement('button');
        this.stopButton.textContent = '⏹️ Stop';
        this.stopButton.style.cssText = `
            padding: 4px 8px;
            background: #666;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.stopButton.onclick = () => this.core.stopPlayback();
        
        // Time display
        this.timeDisplay = document.createElement('span');
        this.timeDisplay.textContent = '00:00.000';
        this.timeDisplay.style.cssText = `
            font-family: monospace;
            color: #fff;
            font-size: 11px;
            margin-left: 8px;
        `;
        
        // Selection display
        this.selectionDisplay = document.createElement('span');
        this.selectionDisplay.textContent = 'No selection';
        this.selectionDisplay.style.cssText = `
            font-family: monospace;
            color: #ffff00;
            font-size: 11px;
            margin-left: 16px;
        `;
        
        playbackContainer.appendChild(this.playButton);
        playbackContainer.appendChild(this.stopButton);
        playbackContainer.appendChild(this.timeDisplay);
        playbackContainer.appendChild(this.selectionDisplay);
        
        this.controls.appendChild(playbackContainer);
    }
    
    createMainControls() {
        // Single row with all main action buttons
        const mainControls = document.createElement('div');
        mainControls.style.cssText = `
            display: flex;
            gap: 6px;
            align-items: center;
            padding: 2px 0;
            flex-wrap: wrap;
        `;
        
        // Button style template
        const buttonStyle = `
            padding: 3px 6px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            color: white;
        `;
        
        // Analyze button
        const analyzeButton = document.createElement('button');
        analyzeButton.textContent = '🔍 Analyze';
        analyzeButton.style.cssText = buttonStyle + 'background: #28a745;';
        analyzeButton.onclick = () => this.core.onParametersChanged();
        
        // Clear selection button
        const clearButton = document.createElement('button');
        clearButton.textContent = '🗑️ Clear';
        clearButton.style.cssText = buttonStyle + 'background: #dc3545;';
        clearButton.onclick = () => this.core.clearSelection();
        
        // Add region button
        const addRegionButton = document.createElement('button');
        addRegionButton.textContent = '➕ Add Region';
        addRegionButton.style.cssText = buttonStyle + 'background: #17a2b8;';
        addRegionButton.onclick = () => this.core.addSelectedRegion();
        
        // Clear all regions button
        const clearAllButton = document.createElement('button');
        clearAllButton.textContent = '🗑️ Clear All';
        clearAllButton.style.cssText = buttonStyle + 'background: #6c757d;';
        clearAllButton.onclick = () => this.core.clearAllRegions();
        
        // Export timing button
        const exportButton = document.createElement('button');
        exportButton.textContent = '📋 Export Timings';
        exportButton.style.cssText = buttonStyle + 'background: #fd7e14;';
        exportButton.onclick = () => this.core.exportTiming();
        
        mainControls.appendChild(analyzeButton);
        mainControls.appendChild(clearButton);
        mainControls.appendChild(addRegionButton);
        mainControls.appendChild(clearAllButton);
        mainControls.appendChild(exportButton);
        
        this.controls.appendChild(mainControls);
    }
    
    createZoomControls() {
        this.zoomControls = document.createElement('div');
        this.zoomControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
            margin-left: auto;
        `;
        
        // Zoom in button
        const zoomInButton = document.createElement('button');
        zoomInButton.textContent = '🔍+';
        zoomInButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomInButton.onclick = () => this.core.zoomIn();
        
        // Zoom out button
        const zoomOutButton = document.createElement('button');
        zoomOutButton.textContent = '🔍-';
        zoomOutButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomOutButton.onclick = () => this.core.zoomOut();
        
        // Reset zoom button
        const resetZoomButton = document.createElement('button');
        resetZoomButton.textContent = '🔄 Reset';
        resetZoomButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        resetZoomButton.onclick = () => this.core.resetZoom();
        
        this.zoomControls.appendChild(zoomInButton);
        this.zoomControls.appendChild(zoomOutButton);
        this.zoomControls.appendChild(resetZoomButton);
        
        // Position zoom controls at the end of the first row
        this.controls.firstChild.appendChild(this.zoomControls);
    }
    
    createStatusDisplays() {
        // Status display
        this.statusDisplay = document.createElement('div');
        this.statusDisplay.style.cssText = `
            font-size: 11px;
            color: #888;
            padding: 2px 0;
            border-top: 1px solid #333;
            margin-top: 4px;
        `;
        this.statusDisplay.textContent = 'Ready to analyze audio';
        
        // Message display
        this.messageDisplay = document.createElement('div');
        this.messageDisplay.style.cssText = `
            font-size: 11px;
            color: #4a9eff;
            padding: 2px 0;
            min-height: 14px;
        `;
        
        this.controls.appendChild(this.statusDisplay);
        this.controls.appendChild(this.messageDisplay);
    }
    
    setupCanvasSize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.core.ctx.scale(devicePixelRatio, devicePixelRatio);
    }
    
    setupCanvasResize() {
        const resizeObserver = new ResizeObserver(() => {
            this.core.resizeCanvas();
        });
        resizeObserver.observe(this.canvas);
    }
    
    setupDragAndDrop() {
        // Add drag and drop functionality
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '0.7';
        });
        
        this.canvas.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '1';
        });
        
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '1';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('audio/')) {
                    
                    // Update the audio_file widget with the file name
                    const audioFileWidget = this.core.node.widgets.find(w => w.name === 'audio_file');
                    if (audioFileWidget) {
                        // For now, just set the name - user will need to provide the actual path
                        audioFileWidget.value = file.name;
                        
                        // Show message that user needs to provide the actual file path
                        this.showMessage(`File dropped: ${file.name}. Please enter the full file path in the audio_file widget.`);
                        
                        // Update the widget display
                        if (audioFileWidget.callback) {
                            audioFileWidget.callback(file.name);
                        }
                    }
                } else {
                    this.showMessage('Please drop an audio file');
                }
            }
        });
    }
    
    // Update time display
    updateTimeDisplay() {
        if (this.timeDisplay) {
            this.timeDisplay.textContent = this.core.formatTime(this.core.currentTime);
        }
    }
    
    // Update selection display
    updateSelectionDisplay() {
        if (this.selectionDisplay) {
            if (this.core.selectedStart !== null && this.core.selectedEnd !== null) {
                const duration = this.core.selectedEnd - this.core.selectedStart;
                this.selectionDisplay.textContent = 
                    `Selected: ${this.core.formatTime(this.core.selectedStart)} - ${this.core.formatTime(this.core.selectedEnd)} (${this.core.formatTime(duration)})`;
            } else {
                this.selectionDisplay.textContent = 'No selection';
            }
        }
    }
    
    // Show message
    showMessage(message) {
        if (this.messageDisplay) {
            this.messageDisplay.textContent = message;
            this.messageDisplay.style.color = '#4a9eff';
            
            // Clear message after 3 seconds
            setTimeout(() => {
                if (this.messageDisplay) {
                    this.messageDisplay.textContent = '';
                }
            }, 3000);
        }
    }
    
    // Update status
    updateStatus(status) {
        if (this.statusDisplay) {
            this.statusDisplay.textContent = status;
        }
    }
    
    resizeNodeForInterface() {
        // Ensure the ComfyUI node is properly sized to contain our 420px interface
        if (!this.core.node) return;
        
        try {
            // Calculate required node dimensions accounting for ComfyUI widget expansion
            const interfaceHeight = 420;
            const interfaceWidth = 800;
            
            // Calculate extra height more accurately (multiline widgets now single-line for testing)
            const widgetCount = this.core.node.widgets ? this.core.node.widgets.length : 8;
            // Account for widgets that will be BELOW our interface (now all single-line)
            const widgetsBelow = Math.max(0, widgetCount - 7); // Assuming 7 widgets before interface
            const nodeExtraHeight = 150 + (widgetsBelow * 25); // Less space needed for single-line widgets
            
            // Set minimum node size with even more generous spacing
            const requiredWidth = Math.max(interfaceWidth, 850);
            const requiredHeight = interfaceHeight + nodeExtraHeight;
            
            if (this.core.node.size) {
                this.core.node.size[0] = Math.max(this.core.node.size[0], requiredWidth);
                this.core.node.size[1] = Math.max(this.core.node.size[1], requiredHeight);
            } else {
                this.core.node.size = [requiredWidth, requiredHeight];
            }
            
            // Force the interface widget itself to have proper height
            if (this.widget) {
                this.widget.computedHeight = interfaceHeight;
                this.widget.height = interfaceHeight;
            }
            
            // Force node to update its layout immediately
            if (this.core.node.onResize) {
                this.core.node.onResize(this.core.node.size);
            }
            
            // Also set min size to prevent shrinking
            if (this.core.node.constructor && this.core.node.constructor.prototype) {
                this.core.node.min_size = [requiredWidth, requiredHeight];
            }
            
            console.log(`🎵 Audio Analyzer: Resized node to ${this.core.node.size[0]}x${this.core.node.size[1]} (widgets: ${widgetCount}, below: ${widgetsBelow})`);
            
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

    repositionWidget(widget) {
        // Move the interface widget to an earlier position to avoid multiline widget overflow
        if (!this.core.node.widgets) return;
        
        try {
            const widgets = this.core.node.widgets;
            const currentIndex = widgets.indexOf(widget);
            
            if (currentIndex === -1) return;
            
            // Store widget values before repositioning to prevent corruption
            const widgetValues = {};
            widgets.forEach((w, i) => {
                if (w.name && w.value !== undefined) {
                    widgetValues[w.name] = w.value;
                }
            });
            
            // Find ideal position - after basic input widgets but before multiline widgets
            let targetIndex = this.findInsertPosition();
            
            // Remove from current position
            widgets.splice(currentIndex, 1);
            
            // Insert at target position
            if (targetIndex > currentIndex) targetIndex--; // Adjust for removal
            widgets.splice(targetIndex, 0, widget);
            
            // Restore widget values after repositioning
            widgets.forEach((w, i) => {
                if (w.name && widgetValues[w.name] !== undefined && w.name !== 'audio_analyzer_interface') {
                    w.value = widgetValues[w.name];
                }
            });
            
            console.log(`🎵 Audio Analyzer: Repositioned interface widget to position ${targetIndex} (was ${currentIndex})`);
            
        } catch (error) {
            console.error('Failed to reposition widget:', error);
        }
    }
    
    findInsertPosition() {
        // Find the best position to insert our interface widget
        if (!this.core.node.widgets) return 0;
        
        const widgets = this.core.node.widgets;
        
        // Look for manual_regions or region_labels widgets (now single-line for testing)
        for (let i = 0; i < widgets.length; i++) {
            const widget = widgets[i];
            if (widget.name === 'manual_regions' || widget.name === 'region_labels') {
                console.log(`🎵 Found region widget '${widget.name}' at position ${i}, inserting interface before it`);
                return i; // Insert before the first region widget
            }
        }
        
        // If no specific region widgets found, try to insert after basic inputs
        // Look for widgets that are likely to be basic inputs
        let insertAfter = 0;
        for (let i = 0; i < widgets.length; i++) {
            const widget = widgets[i];
            // Skip basic input widgets
            if (widget.name && (
                widget.name.includes('audio_file') ||
                widget.name.includes('analysis_method') ||
                widget.name.includes('precision_level') ||
                widget.name.includes('visualization_points') ||
                widget.name.includes('silence_threshold') ||
                widget.name.includes('silence_min_duration') ||
                widget.name.includes('energy_sensitivity')
            )) {
                insertAfter = i + 1;
            } else {
                break; // Stop at first non-basic widget
            }
        }
        
        console.log(`🎵 Inserting interface widget at position ${insertAfter}`);
        return insertAfter;
    }

    setupWidgetHeight(widget) {
        // Ensure the widget properly reserves its height in ComfyUI's layout system
        const interfaceHeight = 420;
        
        // Set multiple height properties to ensure ComfyUI recognizes the widget size
        widget.height = interfaceHeight;
        widget.computedHeight = interfaceHeight;
        widget.last_y = interfaceHeight; // ComfyUI uses this for layout calculations
        
        // Override ComfyUI's height calculation methods
        widget.computeSize = function(width) {
            // Always return our fixed height
            return [width || 780, interfaceHeight];
        };
        
        widget.getHeight = function() {
            return interfaceHeight;
        };
        
        // Add proper draw method to integrate with ComfyUI's rendering system
        widget.draw = function(ctx, node, widget_width, y, widget_height) {
            // This method is called by ComfyUI during node rendering
            // Set the position of our DOM element to match ComfyUI's layout
            if (this.element) {
                const nodeRect = node.getBounding ? node.getBounding() : null;
                
                // Position the element within the node
                this.element.style.position = 'absolute';
                this.element.style.left = '10px'; // Small margin from node edge
                this.element.style.top = (y + 10) + 'px'; // Position at calculated Y + margin
                this.element.style.width = (widget_width - 20) + 'px'; // Width minus margins
                this.element.style.height = interfaceHeight + 'px';
                this.element.style.zIndex = '1000';
                
                // Ensure the element is attached to the node or its parent
                if (!this.element.parentElement || !this.element.parentElement.contains(node.canvas)) {
                    // Find the ComfyUI canvas container
                    const canvasContainer = document.querySelector('.litegraph') || document.body;
                    if (!canvasContainer.contains(this.element)) {
                        canvasContainer.appendChild(this.element);
                    }
                }
            }
        };
        
        // Override mouse handling to ensure proper event coordination
        widget.mouse = function(event, pos, node) {
            // Let the interface handle its own mouse events
            return false; // Don't consume the event
        };
        
        // Ensure the DOM element also has the correct initial styling
        if (widget.element) {
            widget.element.style.height = interfaceHeight + 'px';
            widget.element.style.minHeight = interfaceHeight + 'px';
            widget.element.style.display = 'block';
            widget.element.style.position = 'absolute';
            widget.element.style.boxSizing = 'border-box';
        }
        
        console.log(`🎵 Audio Analyzer: Setup widget height and draw method: ${interfaceHeight}px`);
    }
    
    setupNodeResizeHandling() {
        // Hook into node resize events to keep interface positioned correctly
        if (!this.core.node) return;
        
        // Store original resize method
        const originalOnResize = this.core.node.onResize;
        
        // Override node resize to reposition our interface
        this.core.node.onResize = (size) => {
            // Call original resize if it exists
            if (originalOnResize) {
                originalOnResize.call(this.core.node, size);
            }
            
            // Reposition our interface widget
            this.repositionInterface();
            
            console.log(`🎵 Audio Analyzer: Node resized to ${size[0]}x${size[1]}, repositioning interface`);
        };
        
        // Also hook into the onDrawBackground to ensure interface stays positioned
        const originalOnDrawBackground = this.core.node.onDrawBackground;
        this.core.node.onDrawBackground = function(ctx) {
            // Call original draw background if it exists
            if (originalOnDrawBackground) {
                originalOnDrawBackground.call(this, ctx);
            }
            
            // Trigger interface repositioning
            setTimeout(() => {
                this.audioAnalyzerInterface?.ui?.repositionInterface();
            }, 0);
        };
        
        // Store reference to interface for access in node methods
        this.core.node.audioAnalyzerInterface = this.core;
        
        // Hook into node removal for cleanup
        const originalOnRemoved = this.core.node.onRemoved;
        this.core.node.onRemoved = () => {
            // Call original onRemoved if it exists
            if (originalOnRemoved) {
                originalOnRemoved.call(this.core.node);
            }
            
            // Cleanup our interface
            this.destroy();
        };
        
        // Set up periodic repositioning to handle canvas zoom/pan
        this.positionUpdateInterval = setInterval(() => {
            this.repositionInterface();
        }, 1000); // Update every second
        
        console.log('🎵 Audio Analyzer: Setup node resize handling');
    }
    
    repositionInterface() {
        // Reposition the interface element to stay within node boundaries
        if (!this.widget || !this.widget.element || !this.core.node) return;
        
        try {
            const nodeSize = this.core.node.size;
            const nodePos = this.core.node.pos;
            
            // Find the widget's position within the node layout
            let widgetY = 40; // Start after node title
            if (this.core.node.widgets) {
                for (let i = 0; i < this.core.node.widgets.length; i++) {
                    const widget = this.core.node.widgets[i];
                    if (widget === this.widget) {
                        break; // Found our widget position
                    }
                    // Add height of previous widgets
                    const widgetHeight = widget.computeSize ? widget.computeSize()[1] : 30;
                    widgetY += widgetHeight + 5; // 5px margin between widgets
                }
            }
            
            // Ensure the element is attached to the ComfyUI canvas container
            const canvasContainer = document.querySelector('.litegraph') || document.body;
            if (!canvasContainer.contains(this.widget.element)) {
                canvasContainer.appendChild(this.widget.element);
            }
            
            // Position the interface element using ComfyUI's coordinate system
            this.widget.element.style.position = 'absolute';
            this.widget.element.style.left = (nodePos[0] + 10) + 'px';
            this.widget.element.style.top = (nodePos[1] + widgetY) + 'px';
            this.widget.element.style.width = (nodeSize[0] - 20) + 'px';
            this.widget.element.style.height = '420px';
            this.widget.element.style.zIndex = '1001';
            this.widget.element.style.pointerEvents = 'auto'; // Ensure mouse events work
            
            console.log(`🎵 Interface repositioned to: x=${nodePos[0] + 10}, y=${nodePos[1] + widgetY}, w=${nodeSize[0] - 20}`);
            
        } catch (error) {
            console.error('Failed to reposition interface:', error);
        }
    }
    
    destroy() {
        // Cleanup when the interface is destroyed
        if (this.positionUpdateInterval) {
            clearInterval(this.positionUpdateInterval);
            this.positionUpdateInterval = null;
        }
        
        if (this.widget && this.widget.element && this.widget.element.parentElement) {
            this.widget.element.parentElement.removeChild(this.widget.element);
        }
        
        console.log('🎵 Audio Analyzer UI destroyed and cleaned up');
    }

    ensureUIVisible() {
        // Ensure UI is visible
        
        // Simple approach: let ComfyUI's DOM widget system handle positioning
        // Don't manually append to DOM - let the widget system do it
        if (this.widget && this.widget.element) {
            // The widget system should handle this automatically
            return;
        }
        
        // Only use fallback if widget system failed completely
        setTimeout(() => {
            if (!document.body.contains(this.container)) {
                this.container.style.position = 'relative';
                this.container.style.width = '100%';
                this.container.style.maxWidth = '780px';
                this.container.style.zIndex = '9999';
                document.body.appendChild(this.container);
            }
        }, 200);
    }
}