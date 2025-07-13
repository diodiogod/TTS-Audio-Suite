/**
 * Audio Wave Analyzer Events Module
 * Manages all user interactions and event handling
 */
export class AudioAnalyzerEvents {
    constructor(core) {
        this.core = core;
        this.mouseDown = false;
        this.middleMouseDown = false;
        this.isPanning = false;
        this.isCtrlPanning = false;
        this.lastMousePos = { x: 0, y: 0 };
        
        // Loop marker dragging state
        this.isDraggingLoopStart = false;
        this.isDraggingLoopEnd = false;
        this.loopMarkerHitRadius = 12; // Pixel radius for hit detection
        
        // Amplitude scale dragging state
        this.isDraggingAmplitudeScale = false;
        this.amplitudeDragStartY = 0;
        this.amplitudeDragStartScale = 0;
        
        // Global mouse event handlers for out-of-bounds dragging
        this.globalMouseMoveHandler = null;
        this.globalMouseUpHandler = null;
    }
    
    setupEventListeners() {
        // Mouse events for canvas interaction
        this.core.canvas.addEventListener('mousedown', (e) => {
            // Focus canvas when clicked to enable keyboard shortcuts
            this.core.canvas.focus();
            this.handleMouseDown(e);
        });
        this.core.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.core.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.core.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));
        
        // Wheel event for zooming
        this.core.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        
        // Double-click for time seeking
        this.core.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
        
        // Make canvas focusable and capture keyboard events
        this.core.canvas.tabIndex = 0; // Make canvas focusable
        this.core.canvas.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Focus canvas on mouse enter to ensure keyboard events work
        this.core.canvas.addEventListener('mouseenter', () => {
            this.core.canvas.focus();
        });
        
        // Visual focus indicator (disabled - no blue border)
        this.core.canvas.addEventListener('focus', () => {
            this.core.canvas.style.outline = 'none';
            this.core.showMessage('Audio analyzer focused - keyboard shortcuts active');
        });
        
        this.core.canvas.addEventListener('blur', () => {
            this.core.canvas.style.outline = 'none';
        });
        
        // Prevent context menu on canvas
        this.core.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    }
    
    handleMouseDown(e) {
        if (!this.core.waveformData) return;
        
        // Use coordinate transformation to handle ComfyUI zoom properly
        const coords = this.core.getCanvasCoordinates(e.clientX, e.clientY);
        const time = this.core.pixelToTime(coords.x);
        
        // Debug coordinate transformation (only log occasionally)
        if (!this.lastDebugTime || Date.now() - this.lastDebugTime > 3000) {
            const rect = this.core.canvas.getBoundingClientRect();
            const logicalWidth = this.core.canvas.width / devicePixelRatio;
            // console.log('🎵 Mouse click coordinate transformation:', {  // Debug: coordinate transform (throttled)
            //     client: { x: e.clientX, y: e.clientY },
            //     rect: { 
            //         left: rect.left.toFixed(1), 
            //         top: rect.top.toFixed(1), 
            //         width: rect.width.toFixed(1), 
            //         height: rect.height.toFixed(1) 
            //     },
            //     logical: { width: logicalWidth.toFixed(1) },
            //     scale: { 
            //         x: (rect.width / logicalWidth).toFixed(3), 
            //         devicePixelRatio: devicePixelRatio 
            //     },
            //     canvas: { x: coords.x.toFixed(1), y: coords.y.toFixed(1) },
            //     time: time.toFixed(3) + 's'
            // });
            this.lastDebugTime = Date.now();
        }
        
        this.mouseDown = true;
        this.lastMousePos = coords;
        
        // Check if clicking on amplitude scale first
        if (this.isOverAmplitudeScale(coords.x, coords.y) && e.button === 0) { // Left click on amplitude scale
            this.isDraggingAmplitudeScale = true;
            this.amplitudeDragStartY = coords.y;
            this.amplitudeDragStartScale = this.core.amplitudeScale;
            this.core.canvas.style.cursor = 'ns-resize';
            this.core.showMessage('Dragging amplitude scale - move up/down to resize waveform');
            
            // Start global mouse capture for out-of-bounds dragging
            this.startGlobalAmplitudeCapture(e);
            
            e.preventDefault();
            return;
        }
        
        // Check if clicking on a loop marker next
        const loopMarker = this.getLoopMarkerAtPosition(coords.x, coords.y);
        if (loopMarker && e.button === 0) { // Left click on loop marker
            if (loopMarker === 'start') {
                this.isDraggingLoopStart = true;
                this.core.canvas.style.cursor = 'ew-resize';
                this.core.showMessage('Dragging loop start marker');
            } else if (loopMarker === 'end') {
                this.isDraggingLoopEnd = true;
                this.core.canvas.style.cursor = 'ew-resize';
                this.core.showMessage('Dragging loop end marker');
            }
            e.preventDefault();
            return;
        }
        
        if ((e.button === 0 || e.button === 2) && e.ctrlKey) { // Left or Right + CTRL
            // CTRL + click = panning mode
            this.isCtrlPanning = true;
            this.core.canvas.style.cursor = 'grabbing';
            e.preventDefault(); // Prevent context menu for right click
        } else if (e.button === 0) { // Left mouse button (without CTRL)
            if (e.shiftKey) {
                // Extend selection
                if (this.core.selectedStart !== null) {
                    if (time < this.core.selectedStart) {
                        this.core.selectedEnd = this.core.selectedStart;
                        this.core.selectedStart = time;
                    } else {
                        this.core.selectedEnd = time;
                    }
                    this.core.ui.updateSelectionDisplay();
                    this.core.visualization.redraw();
                }
            } else if (e.altKey) {
                // Alt + left click: Select region for deletion
                const regionIndex = this.core.selectRegionAtTime(time);
                if (regionIndex >= 0) {
                    this.core.showMessage(`Region ${regionIndex + 1} selected. Press Delete key to remove.`);
                } else {
                    this.core.showMessage('No region found at this position.');
                }
            } else {
                // Check if clicking on an existing region first
                const regionIndex = this.core.getRegionAtTime(time);
                if (regionIndex >= 0) {
                    // Left click on existing region: Highlight it (green, persistent)
                    this.core.highlightedRegionIndex = regionIndex;
                    this.core.visualization.redraw();
                    this.core.showMessage(`Region ${regionIndex + 1} highlighted. Press Delete key to remove.`);
                } else {
                    // Start new selection (clear any previous highlight)
                    this.core.highlightedRegionIndex = -1;
                    this.core.isDragging = true;
                    this.core.dragStart = time;
                    this.core.dragEnd = time;
                    this.core.selectedStart = time;
                    this.core.selectedEnd = time;
                    this.core.ui.updateSelectionDisplay();
                    this.core.visualization.redraw();
                }
            }
        } else if (e.button === 1) { // Middle mouse button
            // Start panning
            this.middleMouseDown = true;
            this.isPanning = true;
            this.core.canvas.style.cursor = 'grabbing';
            e.preventDefault(); // Prevent browser's middle-click behavior
        } else if (e.button === 2) { // Right mouse button (without CTRL)
            // Clear selection
            this.core.clearSelection();
        }
    }
    
    handleMouseMove(e) {
        if (!this.core.waveformData) return;
        
        // Use coordinate transformation
        const coords = this.core.getCanvasCoordinates(e.clientX, e.clientY);
        const time = this.core.pixelToTime(coords.x);
        
        if (this.isDraggingLoopStart || this.isDraggingLoopEnd) {
            // Handle loop marker dragging
            const clampedTime = Math.max(0, Math.min(this.core.waveformData.duration, time));
            
            if (this.isDraggingLoopStart) {
                // Ensure start doesn't go past end
                if (this.core.loopEnd !== null && clampedTime < this.core.loopEnd) {
                    this.core.loopStart = clampedTime;
                    this.core.visualization.redraw();
                    this.core.showMessage(`Loop start: ${this.core.formatTime(clampedTime)}`);
                }
            } else if (this.isDraggingLoopEnd) {
                // Ensure end doesn't go before start
                if (this.core.loopStart !== null && clampedTime > this.core.loopStart) {
                    this.core.loopEnd = clampedTime;
                    this.core.visualization.redraw();
                    this.core.showMessage(`Loop end: ${this.core.formatTime(clampedTime)}`);
                }
            }
        } else if (this.mouseDown && this.core.isDragging) {
            // Update drag selection (only when not in CTRL panning mode)
            this.core.dragEnd = time;
            this.core.selectedStart = Math.min(this.core.dragStart, this.core.dragEnd);
            this.core.selectedEnd = Math.max(this.core.dragStart, this.core.dragEnd);
            this.core.ui.updateSelectionDisplay();
            this.core.visualization.redraw();
        } else if (this.isPanning && this.middleMouseDown) {
            // Middle mouse button panning
            const deltaX = coords.x - this.lastMousePos.x;
            const canvasWidth = this.core.canvas.width / devicePixelRatio;
            const visibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
            const timeDelta = -(deltaX / canvasWidth) * visibleDuration;
            
            this.core.scrollOffset = Math.max(0, 
                Math.min(this.core.waveformData.duration - visibleDuration, 
                    this.core.scrollOffset + timeDelta));
            
            this.core.canvas.style.cursor = 'grabbing';
            this.core.visualization.redraw();
        } else if (this.mouseDown && this.isCtrlPanning) {
            // CTRL + left/right drag panning
            const deltaX = coords.x - this.lastMousePos.x;
            const canvasWidth = this.core.canvas.width / devicePixelRatio;
            const visibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
            const timeDelta = -(deltaX / canvasWidth) * visibleDuration;
            
            this.core.scrollOffset = Math.max(0, 
                Math.min(this.core.waveformData.duration - visibleDuration, 
                    this.core.scrollOffset + timeDelta));
            
            this.core.canvas.style.cursor = 'grabbing';
            this.core.visualization.redraw();
        } else if (!this.mouseDown && e.ctrlKey) {
            // Show pan cursor when CTRL is held (regardless of zoom level)
            this.core.canvas.style.cursor = 'grab';
        } else if (!this.mouseDown) {
            // Check if hovering over amplitude scale
            if (this.isOverAmplitudeScale(coords.x, coords.y)) {
                this.core.canvas.style.cursor = 'ns-resize';
            } else {
                // Check if hovering over loop marker
                const loopMarker = this.getLoopMarkerAtPosition(coords.x, coords.y);
                if (loopMarker) {
                    this.core.canvas.style.cursor = 'ew-resize';
                } else {
                    // Default crosshair cursor for precise selection
                    const time = this.core.pixelToTime(coords.x);
                    const regionIndex = this.core.getRegionAtTime(time);
                    
                    if (e.altKey && regionIndex >= 0) {
                        this.core.canvas.style.cursor = 'pointer'; // Show pointer when over region with Alt
                    } else {
                        this.core.canvas.style.cursor = 'crosshair';
                    }
                }
            }
        }
        
        this.lastMousePos = coords;
    }
    
    handleMouseUp(e) {
        if (!this.core.waveformData) return;
        
        this.mouseDown = false;
        
        // Handle amplitude scale drag end (handled by global capture, but keep as fallback)
        if (this.isDraggingAmplitudeScale) {
            this.stopGlobalAmplitudeCapture();
            return;
        }
        
        // Handle loop marker drag end
        if (this.isDraggingLoopStart || this.isDraggingLoopEnd) {
            const markerType = this.isDraggingLoopStart ? 'start' : 'end';
            this.core.showMessage(`Loop ${markerType} marker updated: ${this.core.formatTime(this.isDraggingLoopStart ? this.core.loopStart : this.core.loopEnd)}`);
            this.isDraggingLoopStart = false;
            this.isDraggingLoopEnd = false;
            this.core.canvas.style.cursor = 'crosshair';
            return;
        }
        
        if (e.button === 1) { // Middle mouse button
            this.middleMouseDown = false;
            this.isPanning = false;
            this.core.canvas.style.cursor = 'crosshair';
        }
        
        if (this.isCtrlPanning) {
            this.isCtrlPanning = false;
            // Update cursor based on current state
            if (e.ctrlKey) {
                this.core.canvas.style.cursor = 'grab';
            } else {
                this.core.canvas.style.cursor = 'crosshair';
            }
        }
        
        if (this.core.isDragging) {
            this.core.isDragging = false;
            
            // Ensure selection is valid
            if (Math.abs(this.core.selectedEnd - this.core.selectedStart) < 0.01) {
                // Too small selection, clear it
                this.core.clearSelection();
            } else {
                // Valid selection, update display
                this.core.ui.updateSelectionDisplay();
                this.core.visualization.redraw();
            }
        }
    }
    
    handleMouseLeave(e) {
        this.mouseDown = false;
        this.middleMouseDown = false;
        this.isPanning = false;
        this.isCtrlPanning = false;
        this.isDraggingLoopStart = false;
        this.isDraggingLoopEnd = false;
        
        // DON'T stop amplitude dragging here - let global capture handle it
        // this.isDraggingAmplitudeScale = false;
        
        // Only reset cursor if not amplitude dragging
        if (!this.isDraggingAmplitudeScale) {
            this.core.canvas.style.cursor = 'default';
        }
        
        if (this.core.isDragging) {
            this.core.isDragging = false;
            this.core.ui.updateSelectionDisplay();
            this.core.visualization.redraw();
        }
    }
    
    handleWheel(e) {
        if (!this.core.waveformData) return;
        
        e.preventDefault();
        
        // Use coordinate transformation
        const coords = this.core.getCanvasCoordinates(e.clientX, e.clientY);
        const mouseTime = this.core.pixelToTime(coords.x);
        
        // Zoom in/out based on wheel direction
        const zoomFactor = e.deltaY > 0 ? 0.8 : 1.25;
        const oldZoom = this.core.zoomLevel;
        this.core.zoomLevel = Math.max(0.1, Math.min(100, this.core.zoomLevel * zoomFactor));
        
        // Adjust scroll offset to keep mouse position stable
        if (this.core.zoomLevel !== oldZoom) {
            const canvasWidth = this.core.canvas.width / devicePixelRatio;
            const oldVisibleDuration = this.core.waveformData.duration / oldZoom;
            const newVisibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
            
            const mouseRatio = coords.x / canvasWidth;
            const oldStartTime = this.core.scrollOffset;
            const newStartTime = mouseTime - (mouseRatio * newVisibleDuration);
            
            this.core.scrollOffset = Math.max(0, 
                Math.min(this.core.waveformData.duration - newVisibleDuration, newStartTime));
            
            this.core.visualization.redraw();
        }
    }
    
    handleDoubleClick(e) {
        if (!this.core.waveformData) return;
        
        // Use coordinate transformation
        const coords = this.core.getCanvasCoordinates(e.clientX, e.clientY);
        const time = this.core.pixelToTime(coords.x);
        
        // Seek to clicked position
        this.core.currentTime = Math.max(0, Math.min(this.core.waveformData.duration, time));
        
        if (this.core.audioElement) {
            this.core.audioElement.currentTime = this.core.currentTime;
        }
        
        this.core.ui.updateTimeDisplay();
        this.core.visualization.redraw();
    }
    
    handleKeyDown(e) {
        // Only handle keys when the audio analyzer canvas is focused
        if (!this.core.canvas || document.activeElement !== this.core.canvas) {
            return;
        }
        
        if (!this.core.waveformData) return;
        
        // Prevent ComfyUI from capturing these events
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        switch (e.key) {
            case ' ':
                // Spacebar - toggle playback
                e.preventDefault();
                this.core.togglePlayback();
                break;
                
            case 'Escape':
                // Escape - clear selection
                e.preventDefault();
                this.core.clearSelection();
                break;
                
            case 'Enter':
                // Enter - add selected region
                e.preventDefault();
                this.core.addSelectedRegion();
                break;
                
            case 'Delete':
            case 'Backspace':
                // Delete - delete selected region or clear all if shift held
                e.preventDefault();
                if (e.shiftKey) {
                    this.core.clearAllRegions();
                } else if (this.core.selectedRegionIndices.length > 0 || this.core.highlightedRegionIndex >= 0) {
                    this.core.deleteSelectedRegion();
                } else {
                    this.core.showMessage('No region selected. Click on a region to highlight it for deletion.');
                }
                break;
                
            case 'ArrowLeft':
                // Move playhead left
                e.preventDefault();
                this.core.currentTime = Math.max(0, this.core.currentTime - (e.shiftKey ? 10 : 1));
                if (this.core.audioElement) {
                    this.core.audioElement.currentTime = this.core.currentTime;
                }
                this.core.ui.updateTimeDisplay();
                this.core.visualization.redraw();
                break;
                
            case 'ArrowRight':
                // Move playhead right
                e.preventDefault();
                this.core.currentTime = Math.min(this.core.waveformData.duration, 
                    this.core.currentTime + (e.shiftKey ? 10 : 1));
                if (this.core.audioElement) {
                    this.core.audioElement.currentTime = this.core.currentTime;
                }
                this.core.ui.updateTimeDisplay();
                this.core.visualization.redraw();
                break;
                
            case 'Home':
                // Go to beginning
                e.preventDefault();
                this.core.currentTime = 0;
                if (this.core.audioElement) {
                    this.core.audioElement.currentTime = 0;
                }
                this.core.ui.updateTimeDisplay();
                this.core.visualization.redraw();
                break;
                
            case 'End':
                // Go to end
                e.preventDefault();
                this.core.currentTime = this.core.waveformData.duration;
                if (this.core.audioElement) {
                    this.core.audioElement.currentTime = this.core.currentTime;
                }
                this.core.ui.updateTimeDisplay();
                this.core.visualization.redraw();
                break;
                
            case '+':
            case '=':
                // Zoom in
                e.preventDefault();
                this.core.zoomIn();
                break;
                
            case '-':
                // Zoom out
                e.preventDefault();
                this.core.zoomOut();
                break;
                
            case '0':
                // Reset zoom
                e.preventDefault();
                this.core.resetZoom();
                break;
                
            case 'l':
            case 'L':
                // L - set loop from selection or toggle looping
                e.preventDefault();
                if (e.shiftKey) {
                    this.core.toggleLooping();
                } else {
                    this.core.setLoopFromSelection();
                }
                break;
                
            case 'c':
            case 'C':
                // C - clear loop markers (when shift held)
                if (e.shiftKey) {
                    e.preventDefault();
                    this.core.clearLoopMarkers();
                }
                break;
        }
    }
    
    // Helper method to check if mouse is over a loop marker
    getLoopMarkerAtPosition(x, y) {
        if (!this.core.waveformData || this.core.loopStart === null || this.core.loopEnd === null) {
            return null;
        }
        
        const canvas = this.core.canvas;
        const height = canvas.height / devicePixelRatio;
        const markerHeight = 20;
        const markerY = height - markerHeight;
        
        // Check if y position is in the marker area (bottom area of canvas)
        if (y < markerY - 5 || y > height + 5) {
            return null;
        }
        
        const startX = this.core.timeToPixel(this.core.loopStart);
        const endX = this.core.timeToPixel(this.core.loopEnd);
        
        // Check start marker (triangle with 8px width on each side)
        if (Math.abs(x - startX) <= this.loopMarkerHitRadius) {
            return 'start';
        }
        
        // Check end marker (triangle with 8px width on each side)
        if (Math.abs(x - endX) <= this.loopMarkerHitRadius) {
            return 'end';
        }
        
        return null;
    }
    
    // Helper method to check if mouse is over amplitude scale area (right side)
    isOverAmplitudeScale(x, y) {
        if (!this.core.waveformData) {
            return false;
        }
        
        const canvas = this.core.canvas;
        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;
        
        // Check if mouse is in the right edge area where amplitude labels are drawn
        const amplitudeScaleWidth = 40; // Approximate width of the amplitude scale area
        if (x >= width - amplitudeScaleWidth && x <= width) {
            // Check if y position is within reasonable amplitude label area (not at very top/bottom)
            const marginTop = 20;
            const marginBottom = 20;
            if (y >= marginTop && y <= height - marginBottom) {
                return true;
            }
        }
        
        return false;
    }
    
    // Start global mouse capture for amplitude dragging
    startGlobalAmplitudeCapture(startEvent) {
        // Create global mouse move handler that works anywhere on the document
        this.globalMouseMoveHandler = (e) => {
            if (!this.isDraggingAmplitudeScale) return;
            
            // Get the original canvas coordinates for the starting point
            const startCoords = this.core.getCanvasCoordinates(startEvent.clientX, startEvent.clientY);
            
            // Calculate delta from global mouse position relative to start position
            const deltaY = e.clientY - startEvent.clientY;
            const sensitivity = 0.002; // Same sensitivity as before
            
            // Calculate new amplitude scale (drag up = bigger waves, drag down = smaller waves)
            const newScale = this.amplitudeDragStartScale - (deltaY * sensitivity);
            
            // Clamp amplitude scale between reasonable limits
            this.core.amplitudeScale = Math.max(0.05, Math.min(1.5, newScale));
            
            this.core.visualization.redraw();
            this.core.showMessage(`Amplitude scale: ${(this.core.amplitudeScale / 0.4).toFixed(2)}x`);
            
            // Prevent default to avoid any unwanted behaviors
            e.preventDefault();
        };
        
        // Create global mouse up handler
        this.globalMouseUpHandler = (e) => {
            if (!this.isDraggingAmplitudeScale) return;
            
            // Stop amplitude dragging
            this.core.showMessage(`Amplitude scale set to: ${(this.core.amplitudeScale / 0.4).toFixed(2)}x`);
            this.stopGlobalAmplitudeCapture();
            
            e.preventDefault();
        };
        
        // Add global event listeners
        document.addEventListener('mousemove', this.globalMouseMoveHandler, { passive: false });
        document.addEventListener('mouseup', this.globalMouseUpHandler, { passive: false });
    }
    
    // Stop global mouse capture for amplitude dragging
    stopGlobalAmplitudeCapture() {
        this.isDraggingAmplitudeScale = false;
        this.core.canvas.style.cursor = 'crosshair';
        
        // Remove global event listeners
        if (this.globalMouseMoveHandler) {
            document.removeEventListener('mousemove', this.globalMouseMoveHandler);
            this.globalMouseMoveHandler = null;
        }
        
        if (this.globalMouseUpHandler) {
            document.removeEventListener('mouseup', this.globalMouseUpHandler);
            this.globalMouseUpHandler = null;
        }
    }
    
    // Cleanup method to remove global listeners (call when destroying component)
    cleanup() {
        this.stopGlobalAmplitudeCapture();
    }
}