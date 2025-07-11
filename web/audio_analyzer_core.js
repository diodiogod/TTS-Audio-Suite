import { app } from "../../scripts/app.js";
import { AudioAnalyzerUI } from "./audio_analyzer_ui.js";
import { AudioAnalyzerEvents } from "./audio_analyzer_events.js";
import { AudioAnalyzerVisualization } from "./audio_analyzer_visualization.js";
import { AudioAnalyzerNodeIntegration } from "./audio_analyzer_node_integration.js";

/**
 * Core Audio Analyzer Interface
 * Main class that coordinates all audio analyzer functionality
 */
export class AudioAnalyzerInterface {
    constructor(node) {
        this.node = node;
        this.canvas = null;
        this.ctx = null;
        this.waveformData = null;
        this.selectedRegions = [];
        this.zoomLevel = 1;
        this.scrollOffset = 0;
        this.isPlaying = false;
        this.currentTime = 0;
        this.audioElement = null;
        this.isDragging = false;
        this.dragStart = null;
        this.dragEnd = null;
        this.selectedStart = null;
        this.selectedEnd = null;
        this.selectedRegionIndex = -1; // For tracking which region is selected for deletion
        this.hoveredRegionIndex = -1; // For visual feedback
        
        // Loop markers
        this.loopStart = null;
        this.loopEnd = null;
        this.isLooping = false;
        
        // Color scheme for the interface
        this.colors = {
            background: '#1a1a1a',
            waveform: '#4a9eff',
            rms: '#ff6b6b',
            grid: '#333333',
            selection: 'rgba(255, 255, 0, 0.3)',
            playhead: '#ff0000',
            region: 'rgba(0, 255, 0, 0.2)',
            regionSelected: 'rgba(255, 165, 0, 0.4)',
            regionHovered: 'rgba(0, 255, 0, 0.4)',
            loopMarker: '#ff00ff',
            text: '#ffffff'
        };
        
        // Initialize modules
        this.ui = new AudioAnalyzerUI(this);
        this.events = new AudioAnalyzerEvents(this);
        this.visualization = new AudioAnalyzerVisualization(this);
        this.nodeIntegration = new AudioAnalyzerNodeIntegration(this);
        
        this.setupInterface();
    }
    
    setupInterface() {
        // Create the main interface using UI module
        this.ui.createInterface();
        
        // Setup event listeners
        this.events.setupEventListeners();
        
        // Setup canvas resize observer
        this.ui.setupCanvasResize();
        
        // Show initial message
        this.visualization.showInitialMessage();
    }
    
    // Utility functions
    pixelToTime(pixel) {
        if (!this.waveformData) return 0;
        
        const canvasWidth = this.canvas.width / devicePixelRatio;
        const visibleDuration = this.waveformData.duration / this.zoomLevel;
        const startTime = this.scrollOffset;
        
        return startTime + (pixel / canvasWidth) * visibleDuration;
    }
    
    timeToPixel(time) {
        if (!this.waveformData) return 0;
        
        const canvasWidth = this.canvas.width / devicePixelRatio;
        const visibleDuration = this.waveformData.duration / this.zoomLevel;
        const startTime = this.scrollOffset;
        
        return ((time - startTime) / visibleDuration) * canvasWidth;
    }
    
    // Coordinate transformation utilities for ComfyUI zoom handling
    getCanvasCoordinates(clientX, clientY) {
        if (!this.canvas) return { x: 0, y: 0 };
        
        // Get the canvas bounding rect (this is already in viewport coordinates)
        const rect = this.canvas.getBoundingClientRect();
        
        // Calculate coordinates relative to canvas
        // getBoundingClientRect already accounts for all CSS transforms including ComfyUI zoom
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        
        // Don't apply additional transformation - getBoundingClientRect handles it
        return { x, y };
    }
    
    getComfyUIZoomLevel() {
        // Try to get ComfyUI's zoom level from various possible sources
        try {
            // Method 1: Check for app canvas zoom
            if (window.app && window.app.canvas && typeof window.app.canvas.ds !== 'undefined') {
                return window.app.canvas.ds.scale || 1;
            }
            
            // Method 2: Check for LiteGraph canvas zoom  
            if (window.LiteGraph && window.LiteGraph.LGraphCanvas && window.LiteGraph.LGraphCanvas.active_canvas) {
                const canvas = window.LiteGraph.LGraphCanvas.active_canvas;
                return canvas.ds?.scale || 1;
            }
            
            // Method 3: Check node graph zoom
            if (this.node && this.node.graph && this.node.graph.canvas) {
                return this.node.graph.canvas.ds?.scale || 1;
            }
            
            // Method 4: Check for ComfyUI main canvas element and its transform
            const comfyCanvas = document.querySelector('.litegraph canvas');
            if (comfyCanvas) {
                const style = window.getComputedStyle(comfyCanvas);
                const transform = style.transform;
                if (transform && transform !== 'none') {
                    const match = transform.match(/scale\(([^)]+)\)/);
                    if (match) {
                        return parseFloat(match[1]) || 1;
                    }
                }
            }
            
            return 1; // Default to no zoom if we can't detect it
        } catch (error) {
            console.warn('Failed to get ComfyUI zoom level:', error);
            return 1;
        }
    }
    
    getComfyUITransform() {
        // Get the full transformation matrix including zoom and pan offset
        try {
            // Method 1: Check for app canvas transform
            if (window.app && window.app.canvas && window.app.canvas.ds) {
                const ds = window.app.canvas.ds;
                return {
                    scale: ds.scale || 1,
                    offset: { x: ds.offset?.[0] || 0, y: ds.offset?.[1] || 0 }
                };
            }
            
            // Method 2: Check for LiteGraph canvas transform
            if (window.LiteGraph && window.LiteGraph.LGraphCanvas && window.LiteGraph.LGraphCanvas.active_canvas) {
                const canvas = window.LiteGraph.LGraphCanvas.active_canvas;
                const ds = canvas.ds;
                if (ds) {
                    return {
                        scale: ds.scale || 1,
                        offset: { x: ds.offset?.[0] || 0, y: ds.offset?.[1] || 0 }
                    };
                }
            }
            
            // Method 3: Check node graph transform
            if (this.node && this.node.graph && this.node.graph.canvas && this.node.graph.canvas.ds) {
                const ds = this.node.graph.canvas.ds;
                return {
                    scale: ds.scale || 1,
                    offset: { x: ds.offset?.[0] || 0, y: ds.offset?.[1] || 0 }
                };
            }
            
            return null; // No transform available
        } catch (error) {
            console.warn('Failed to get ComfyUI transform:', error);
            return null;
        }
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
    }
    
    // Canvas management
    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.ctx.scale(devicePixelRatio, devicePixelRatio);
        this.visualization.redraw();
    }
    
    // Selection management
    setSelection(startTime, endTime) {
        this.selectedStart = startTime;
        this.selectedEnd = endTime;
        this.ui.updateSelectionDisplay();
        this.visualization.redraw();
    }
    
    
    // Region management
    addSelectedRegion() {
        if (this.selectedStart !== null && this.selectedEnd !== null) {
            const region = {
                start: this.selectedStart,
                end: this.selectedEnd,
                label: `Region ${this.selectedRegions.length + 1}`,
                id: Date.now()
            };
            
            this.selectedRegions.push(region);
            this.clearSelection();
            this.visualization.redraw();
            
            // Update manual regions in the node
            this.updateManualRegions();
        }
    }
    
    clearAllRegions() {
        this.selectedRegions = [];
        this.selectedRegionIndex = -1;
        this.hoveredRegionIndex = -1;
        this.visualization.redraw();
        this.updateManualRegions();
    }
    
    deleteSelectedRegion() {
        if (this.selectedRegionIndex >= 0 && this.selectedRegionIndex < this.selectedRegions.length) {
            this.selectedRegions.splice(this.selectedRegionIndex, 1);
            this.selectedRegionIndex = -1;
            this.hoveredRegionIndex = -1;
            // Renumber remaining regions
            this.selectedRegions.forEach((region, index) => {
                region.label = `Region ${index + 1}`;
            });
            this.visualization.redraw();
            this.updateManualRegions();
        }
    }
    
    selectRegionAtTime(time) {
        // Find which region contains this time
        for (let i = 0; i < this.selectedRegions.length; i++) {
            const region = this.selectedRegions[i];
            if (time >= region.start && time <= region.end) {
                this.selectedRegionIndex = i;
                this.visualization.redraw();
                return i;
            }
        }
        this.selectedRegionIndex = -1;
        this.visualization.redraw();
        return -1;
    }
    
    getRegionAtTime(time) {
        for (let i = 0; i < this.selectedRegions.length; i++) {
            const region = this.selectedRegions[i];
            if (time >= region.start && time <= region.end) {
                return i;
            }
        }
        return -1;
    }
    
    updateManualRegions() {
        // Update the manual_regions widget with current selections (multiline format)
        const manualRegionsWidget = this.node.widgets.find(w => w.name === 'manual_regions');
        if (manualRegionsWidget) {
            const regionsText = this.selectedRegions
                .map(r => `${r.start.toFixed(3)},${r.end.toFixed(3)}`)
                .join('\n'); // Use newline separator for multiline widget
            manualRegionsWidget.value = regionsText;
        }
        
        // Update labels widget (multiline format)
        const labelsWidget = this.node.widgets.find(w => w.name === 'region_labels');
        if (labelsWidget) {
            const labelsText = this.selectedRegions
                .map(r => r.label)
                .join('\n'); // Use newline separator for multiline widget
            labelsWidget.value = labelsText;
        }
    }
    
    // Loop marker management
    setLoopFromSelection() {
        if (this.selectedStart !== null && this.selectedEnd !== null) {
            this.loopStart = this.selectedStart;
            this.loopEnd = this.selectedEnd;
            this.visualization.redraw();
            this.showMessage(`Loop set: ${this.formatTime(this.loopStart)} - ${this.formatTime(this.loopEnd)}`);
        } else {
            this.showMessage('Please select a region first to set loop markers');
        }
    }
    
    setLoopFromRegion(regionIndex) {
        if (regionIndex >= 0 && regionIndex < this.selectedRegions.length) {
            const region = this.selectedRegions[regionIndex];
            this.loopStart = region.start;
            this.loopEnd = region.end;
            this.visualization.redraw();
            this.showMessage(`Loop set from ${region.label}: ${this.formatTime(this.loopStart)} - ${this.formatTime(this.loopEnd)}`);
        }
    }
    
    clearLoopMarkers() {
        this.loopStart = null;
        this.loopEnd = null;
        this.isLooping = false;
        this.visualization.redraw();
        this.showMessage('Loop markers cleared');
    }
    
    toggleLooping() {
        this.isLooping = !this.isLooping;
        this.showMessage(this.isLooping ? 'Looping enabled' : 'Looping disabled');
    }
    
    // Audio playback controls
    togglePlayback() {
        if (this.isPlaying) {
            this.pausePlayback();
        } else {
            this.startPlayback();
        }
    }
    
    startPlayback() {
        if (!this.audioElement) return;
        
        // If looping is enabled and we have loop markers, start from loop start
        if (this.isLooping && this.loopStart !== null) {
            this.currentTime = this.loopStart;
        }
        
        this.audioElement.currentTime = this.currentTime;
        this.audioElement.play();
        this.isPlaying = true;
        this.ui.playButton.textContent = '⏸️ Pause';
        
        // Update playhead position
        this.updatePlayhead();
    }
    
    pausePlayback() {
        if (this.audioElement) {
            this.audioElement.pause();
        }
        this.isPlaying = false;
        this.stopPlayheadAnimation(); // Stop animation loop
        this.ui.playButton.textContent = '▶️ Play';
    }
    
    stopPlayback() {
        if (this.audioElement) {
            this.audioElement.pause();
            this.audioElement.currentTime = 0;
        }
        this.isPlaying = false;
        this.currentTime = 0;
        this.stopPlayheadAnimation(); // Stop animation loop
        this.visualization.stopAnimation(); // Stop visualization animation loop
        this.ui.playButton.textContent = '▶️ Play';
        this.ui.updateTimeDisplay();
        this.visualization.redraw();
    }
    
    updatePlayhead() {
        // Double-check both isPlaying and audio element state
        if (!this.isPlaying || !this.audioElement || this.audioElement.ended || this.audioElement.paused) {
            return;
        }
        
        if (this.audioElement) {
            this.currentTime = this.audioElement.currentTime;
            
            // Check for loop end
            if (this.isLooping && this.loopEnd !== null && this.currentTime >= this.loopEnd) {
                this.currentTime = this.loopStart || 0;
                this.audioElement.currentTime = this.currentTime;
            }
            
            this.ui.updateTimeDisplay();
            this.visualization.redraw();
        }
        
        // Triple-check before scheduling next frame
        if (this.isPlaying && this.audioElement && !this.audioElement.ended && !this.audioElement.paused) {
            this.playheadAnimationId = requestAnimationFrame(() => this.updatePlayhead());
        }
    }
    
    stopPlayheadAnimation() {
        if (this.playheadAnimationId) {
            cancelAnimationFrame(this.playheadAnimationId);
            this.playheadAnimationId = null;
        }
    }
    
    // Zoom controls
    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 2, 100);
        this.visualization.redraw();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 2, 0.1);
        this.visualization.redraw();
    }
    
    resetZoom() {
        this.zoomLevel = 1;
        this.scrollOffset = 0;
        this.visualization.redraw();
    }
    
    // Export functionality
    exportTiming() {
        if (this.selectedRegions.length === 0) {
            alert('No regions selected. Please select timing regions first.');
            return;
        }
        
        const timingData = this.selectedRegions
            .map(r => `${r.start.toFixed(3)},${r.end.toFixed(3)}`)
            .join('\n');
        
        // Copy to clipboard
        navigator.clipboard.writeText(timingData).then(() => {
            alert('Timing data copied to clipboard!');
        }).catch(() => {
            // Fallback: show in alert
            alert(`Timing data:\n${timingData}`);
        });
    }
    
    // Show message
    showMessage(message) {
        this.ui.showMessage(message);
    }
    
    // Update visualization with new data
    updateVisualization(data) {
        this.nodeIntegration.updateVisualization(data);
    }
    
    // Handle audio file selection
    onAudioFileSelected(filePath) {
        this.nodeIntegration.onAudioFileSelected(filePath);
    }
    
    // Handle parameter changes
    onParametersChanged() {
        this.nodeIntegration.onParametersChanged();
    }
    
    // Handle audio connection
    onAudioConnected() {
        this.nodeIntegration.onAudioConnected();
    }
    
    // Clear current selection (but don't clear region selection)
    clearSelection() {
        this.selectedStart = null;
        this.selectedEnd = null;
        this.dragStart = null;
        this.dragEnd = null;
        this.isDragging = false;
        this.ui.updateSelectionDisplay();
        this.visualization.redraw();
    }
}