/**
 * Audio Analyzer Events Module
 * Manages all user interactions and event handling
 */
export class AudioAnalyzerEvents {
    constructor(core) {
        this.core = core;
        this.mouseDown = false;
        this.lastMousePos = { x: 0, y: 0 };
    }
    
    setupEventListeners() {
        // Mouse events for canvas interaction
        this.core.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.core.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.core.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.core.canvas.addEventListener('mouseleave', (e) => this.handleMouseLeave(e));
        
        // Wheel event for zooming
        this.core.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        
        // Double-click for time seeking
        this.core.canvas.addEventListener('dblclick', (e) => this.handleDoubleClick(e));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
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
            console.log('🎵 Mouse click:', {
                client: { x: e.clientX, y: e.clientY },
                rect: { left: rect.left.toFixed(1), top: rect.top.toFixed(1), width: rect.width.toFixed(1) },
                canvas: coords,
                time: time.toFixed(3) + 's'
            });
            this.lastDebugTime = Date.now();
        }
        
        this.mouseDown = true;
        this.lastMousePos = coords;
        
        if (e.button === 0) { // Left mouse button
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
            } else {
                // Start new selection
                this.core.isDragging = true;
                this.core.dragStart = time;
                this.core.dragEnd = time;
                this.core.selectedStart = time;
                this.core.selectedEnd = time;
                this.core.ui.updateSelectionDisplay();
                this.core.visualization.redraw();
            }
        } else if (e.button === 2) { // Right mouse button
            // Clear selection
            this.core.clearSelection();
        }
    }
    
    handleMouseMove(e) {
        if (!this.core.waveformData) return;
        
        // Use coordinate transformation
        const coords = this.core.getCanvasCoordinates(e.clientX, e.clientY);
        const time = this.core.pixelToTime(coords.x);
        
        if (this.mouseDown && this.core.isDragging) {
            // Update drag selection
            this.core.dragEnd = time;
            this.core.selectedStart = Math.min(this.core.dragStart, this.core.dragEnd);
            this.core.selectedEnd = Math.max(this.core.dragStart, this.core.dragEnd);
            this.core.ui.updateSelectionDisplay();
            this.core.visualization.redraw();
        } else if (this.mouseDown && e.ctrlKey) {
            // Pan the view
            const deltaX = coords.x - this.lastMousePos.x;
            const canvasWidth = this.core.canvas.width / devicePixelRatio;
            const visibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
            const timeDelta = -(deltaX / canvasWidth) * visibleDuration;
            
            this.core.scrollOffset = Math.max(0, 
                Math.min(this.core.waveformData.duration - visibleDuration, 
                    this.core.scrollOffset + timeDelta));
            
            this.core.visualization.redraw();
        }
        
        this.lastMousePos = coords;
    }
    
    handleMouseUp(e) {
        if (!this.core.waveformData) return;
        
        this.mouseDown = false;
        
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
        // Only handle keys when the audio analyzer is focused
        if (!this.core.canvas || document.activeElement !== this.core.canvas) {
            return;
        }
        
        if (!this.core.waveformData) return;
        
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
                // Delete - clear all regions
                e.preventDefault();
                this.core.clearAllRegions();
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
        }
    }
}