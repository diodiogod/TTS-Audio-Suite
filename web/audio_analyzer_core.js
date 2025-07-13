import { app } from "../../scripts/app.js";
import { AudioAnalyzerUI } from "./audio_analyzer_ui.js";
import { AudioAnalyzerEvents } from "./audio_analyzer_events.js";
import { AudioAnalyzerVisualization } from "./audio_analyzer_visualization.js";
import { AudioAnalyzerNodeIntegration } from "./audio_analyzer_node_integration.js";

/**
 * Core Audio Wave Analyzer Interface
 * Main class that coordinates all audio wave analyzer functionality
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
        this.playbackSpeed = 1;
        this.isDragging = false;
        this.dragStart = null;
        this.dragEnd = null;
        this.selectedStart = null;
        this.selectedEnd = null;
        this.selectedRegionIndices = []; // For tracking multiple regions selected for deletion
        this.highlightedRegionIndex = -1; // For click-to-highlight feedback (persists until next click)
        
        // Loop markers
        this.loopStart = null;
        this.loopEnd = null;
        this.isLooping = false;
        
        // Amplitude scaling
        this.amplitudeScale = 0.4; // Default scaling factor (0.1 to 1.0)
        this.maxAmplitudeRange = 1.0; // Maximum amplitude range (+/- 1.0)
        
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
        
        // Setup widget change listeners for bidirectional sync
        this.setupWidgetListeners();
        
        // Attempt to load cached data, otherwise show initial message
        this.loadCachedData();
    }
    
    setupWidgetListeners() {
        // Store the last known values for comparison
        const manualRegionsWidget = this.node.widgets.find(w => w.name === 'manual_regions');
        const labelsWidget = this.node.widgets.find(w => w.name === 'region_labels');
        
        if (manualRegionsWidget) {
            this.lastManualRegionsValue = manualRegionsWidget.value;
        }
        if (labelsWidget) {
            this.lastLabelsValue = labelsWidget.value;
        }
        
        // Add canvas click listener to check for text changes
        this.canvas.addEventListener('mousedown', () => {
            this.checkAndSyncTextChanges();
        });
        
        // Also check when canvas gets focus (user clicks on interface)
        this.canvas.addEventListener('focus', () => {
            this.checkAndSyncTextChanges();
        });
    }
    
    checkAndSyncTextChanges() {
        const manualRegionsWidget = this.node.widgets.find(w => w.name === 'manual_regions');
        const labelsWidget = this.node.widgets.find(w => w.name === 'region_labels');
        
        let hasChanges = false;
        
        // Check if manual_regions text changed
        if (manualRegionsWidget && manualRegionsWidget.value !== this.lastManualRegionsValue) {
            this.lastManualRegionsValue = manualRegionsWidget.value;
            hasChanges = true;
        }
        
        // Check if labels text changed
        if (labelsWidget && labelsWidget.value !== this.lastLabelsValue) {
            this.lastLabelsValue = labelsWidget.value;
            hasChanges = true;
        }
        
        // Only sync if there were actual changes
        if (hasChanges && !manualRegionsWidget?._updating) {
            this.parseManualRegionsFromText();
            this.showMessage('Regions synced from text input');
        }
    }
    
    async loadCachedData() {
        // Attempts to load persisted analysis data from the last execution.
        const cacheUrl = `/output/audio_analyzer_cache_${this.node.id}.json?t=${Date.now()}`;
        try {
            const response = await fetch(cacheUrl);
            if (!response.ok) {
                throw new Error(`Cache file not found or server error: ${response.status}`);
            }
            const data = await response.json();
            this.nodeIntegration.updateVisualization(data);
            console.log(`🌊 Audio Wave Analyzer: Successfully loaded cached data for node ${this.node.id}`);
        } catch (error) {
            // console.log(`🌊 Audio Wave Analyzer: No cache found for node ${this.node.id}. Showing initial message. Details: ${error.message}`);  // Debug: cache miss
            this.visualization.showInitialMessage();
        }
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
        
        // Get the canvas bounding rect (this gives display size affected by ComfyUI zoom)
        const rect = this.canvas.getBoundingClientRect();
        
        // Calculate coordinates relative to canvas in display pixels
        const displayX = clientX - rect.left;
        const displayY = clientY - rect.top;
        
        // Convert from display pixels to logical canvas pixels
        // Canvas logical size is canvas.width / devicePixelRatio
        const logicalCanvasWidth = this.canvas.width / devicePixelRatio;
        const logicalCanvasHeight = this.canvas.height / devicePixelRatio;
        
        // Scale coordinates from display size to logical size
        const x = (displayX / rect.width) * logicalCanvasWidth;
        const y = (displayY / rect.height) * logicalCanvasHeight;
        
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
    
    getPrecisionMode() {
        const precisionWidget = this.node.widgets?.find(w => w.name === 'precision_level');
        return precisionWidget ? precisionWidget.value : 'miliseconds'; // Default to 'miliseconds'
    }

    formatTime(seconds) {
        const mode = this.getPrecisionMode();

        if (mode === 'Samples') {
            if (!this.waveformData || !this.waveformData.sampleRate) {
                return 'N/A Samples';
            }
            const sampleIndex = Math.floor(seconds * this.waveformData.sampleRate);
            return `${sampleIndex.toLocaleString()} smp`;
        }

        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);

        if (mode === 'seconds') {
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }

        // Default to 'miliseconds'
        const ms = Math.floor((seconds % 1) * 1000);
        const msString = ms.toString().padStart(3, '0');
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${msString}`;
    }
    
    formatRegionValue(timeInSeconds) {
        const mode = this.getPrecisionMode();

        switch (mode) {
            case 'Samples':
                if (!this.waveformData || !this.waveformData.sampleRate) return '0';
                return Math.floor(timeInSeconds * this.waveformData.sampleRate).toString();
            case 'seconds':
                return Math.round(timeInSeconds).toString(); // Use Math.round for whole seconds
            case 'miliseconds':
            default:
                return timeInSeconds.toFixed(3);
        }
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
            
            // Sort regions chronologically by start time
            this.selectedRegions.sort((a, b) => a.start - b.start);
            
            // Renumber regions to be chronological
            this.selectedRegions.forEach((region, index) => {
                if (region.label.match(/^Region \d+$/)) {
                    region.label = `Region ${index + 1}`;
                }
            });
            
            this.clearSelection();
            this.visualization.redraw();
            
            // Update manual regions in the node
            this.updateManualRegions();
        }
    }
    
    clearAllRegions() {
        this.selectedRegions = [];
        this.selectedRegionIndices = [];
        this.highlightedRegionIndex = -1;
        
        // Only clear manual regions, keep auto-detected regions
        // (waveformData.regions only contains auto-detected regions after fix above)
        
        this.visualization.redraw();
        this.updateManualRegions();
    }
    
    deleteSelectedRegion() {
        if (this.selectedRegionIndices.length > 0) {
            // Delete multiple selected regions (orange) - sort indices in reverse order to avoid index shifting
            const sortedIndices = [...this.selectedRegionIndices].sort((a, b) => b - a);
            sortedIndices.forEach(index => {
                if (index >= 0 && index < this.selectedRegions.length) {
                    this.selectedRegions.splice(index, 1);
                }
            });
            this.selectedRegionIndices = [];
            this.showMessage(`Deleted ${sortedIndices.length} selected regions`);
        } else if (this.highlightedRegionIndex >= 0 && this.highlightedRegionIndex < this.selectedRegions.length) {
            // Delete single hovered region (green)
            this.selectedRegions.splice(this.highlightedRegionIndex, 1);
            this.highlightedRegionIndex = -1;
            this.showMessage('Deleted hovered region');
        }
        
        // Sort regions chronologically by start time (in case they weren't)
        this.selectedRegions.sort((a, b) => a.start - b.start);
        
        // Renumber remaining regions chronologically
        this.selectedRegions.forEach((region, index) => {
            if (region.label.match(/^Region \d+$/)) {
                region.label = `Region ${index + 1}`;
            }
        });
        
        this.visualization.redraw();
        this.updateManualRegions();
    }
    
    selectRegionAtTime(time) {
        // Find which region contains this time
        for (let i = 0; i < this.selectedRegions.length; i++) {
            const region = this.selectedRegions[i];
            if (time >= region.start && time <= region.end) {
                // Toggle selection: if already selected, deselect it; otherwise add it
                const existingIndex = this.selectedRegionIndices.indexOf(i);
                if (existingIndex >= 0) {
                    // Deselect: remove from array
                    this.selectedRegionIndices.splice(existingIndex, 1);
                    this.showMessage(`Region ${i + 1} deselected. ${this.selectedRegionIndices.length} regions selected.`);
                } else {
                    // Select: add to array
                    this.selectedRegionIndices.push(i);
                    this.showMessage(`Region ${i + 1} selected. ${this.selectedRegionIndices.length} regions selected.`);
                }
                this.visualization.redraw();
                return i;
            }
        }
        // Clicked empty space - clear all selections
        this.selectedRegionIndices = [];
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
            // Temporarily disable the change listener to prevent infinite loop
            manualRegionsWidget._updating = true;
            const regionsText = this.selectedRegions
                .map(r => `${this.formatRegionValue(r.start)},${this.formatRegionValue(r.end)}`)
                .join('\n'); // Use newline separator for multiline widget
            manualRegionsWidget.value = regionsText;
            manualRegionsWidget._updating = false;
        }
        
        // Update labels widget (multiline format)
        const labelsWidget = this.node.widgets.find(w => w.name === 'region_labels');
        if (labelsWidget) {
            // Temporarily disable the change listener to prevent infinite loop
            labelsWidget._updating = true;
            const labelsText = this.selectedRegions
                .map(r => r.label) // Labels don't need precision, but we keep the structure
                .join('\n'); // Use newline separator for multiline widget
            labelsWidget.value = labelsText;
            labelsWidget._updating = false;
        }
    }
    
    // Parse manual regions text back into selectedRegions array
    parseManualRegionsFromText() {
        const manualRegionsWidget = this.node.widgets.find(w => w.name === 'manual_regions');
        const labelsWidget = this.node.widgets.find(w => w.name === 'region_labels');
        
        if (!manualRegionsWidget || !manualRegionsWidget.value.trim()) {
            this.selectedRegions = [];
            this.visualization.redraw();
            return;
        }
        
        const regionsText = manualRegionsWidget.value.trim();
        const labelsText = labelsWidget ? labelsWidget.value.trim() : '';
        
        const regionLines = regionsText.split('\n').filter(line => line.trim());
        const labelLines = labelsText ? labelsText.split('\n').filter(line => line.trim()) : [];
        
        const newRegions = [];
        
        regionLines.forEach((line, index) => {
            const trimmed = line.trim();
            if (!trimmed) return;
            
            // Parse start,end format
            const parts = trimmed.split(',').map(p => p.trim());
            if (parts.length === 2) {
                const start = parseFloat(parts[0]);
                const end = parseFloat(parts[1]);
                
                if (!isNaN(start) && !isNaN(end) && start < end) {
                    const label = labelLines[index] || `Region ${index + 1}`;
                    newRegions.push({
                        start: start,
                        end: end,
                        label: label,
                        id: Date.now() + index // Unique ID
                    });
                }
            }
        });
        
        // Sort regions chronologically by start time
        newRegions.sort((a, b) => a.start - b.start);
        
        // Update labels to be chronological (Region 1, Region 2, etc.)
        newRegions.forEach((region, index) => {
            if (region.label.match(/^Region \d+$/)) {
                region.label = `Region ${index + 1}`;
            }
        });
        
        this.selectedRegions = newRegions;
        this.visualization.redraw();
        
        // Update the labels widget to reflect any changes
        this.updateManualRegions();
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
        if (!this.isPlaying) {
            return;
        }
        
        if (this.playbackSpeed < 0) {
            // Backwards movement: manual time control, silent audio
            const deltaTime = Math.abs(this.playbackSpeed) / 60; // Approximate frame rate adjustment
            this.currentTime -= deltaTime;
            
            // Clamp to audio bounds and stop at start (don't loop)
            if (this.currentTime < 0) {
                this.currentTime = 0;
                // Don't stop playing - just clamp position
            }
            
            // Handle looping in reverse
            if (this.isLooping && this.loopStart !== null && this.currentTime <= this.loopStart) {
                this.currentTime = this.loopEnd || (this.waveformData ? this.waveformData.duration : 0);
            }
        } else {
            // Normal forward playback
            if (this.audioElement) {
                if (this.audioElement.ended) {
                    // Audio ended but we're still "playing" - clamp to end position
                    this.currentTime = this.waveformData ? this.waveformData.duration : this.audioElement.duration;
                    // Don't stop playing - just clamp position
                } else if (!this.audioElement.paused) {
                    this.currentTime = this.audioElement.currentTime;
                    
                    // Check for loop end
                    if (this.isLooping && this.loopEnd !== null && this.currentTime >= this.loopEnd) {
                        this.currentTime = this.loopStart || 0;
                        this.audioElement.currentTime = this.currentTime;
                    }
                }
            }
        }
        
        this.ui.updateTimeDisplay();
        this.visualization.redraw();
        
        // Continue animation if still playing
        if (this.isPlaying) {
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
        this.amplitudeScale = 0.4; // Reset amplitude scale to default
        
        // Reset speed to 1x
        this.playbackSpeed = 1;
        if (this.ui.speedSlider) {
            this.ui.speedSlider.value = '1';
        }
        if (this.ui.speedValue) {
            this.ui.speedValue.textContent = '1.00x';
        }
        if (this.audioElement) {
            this.audioElement.playbackRate = 1;
            if (this.audioElement.volume === 0) {
                this.audioElement.volume = 1; // Restore volume if muted from backwards mode
            }
        }
        
        this.visualization.redraw();
    }
    
    // Speed control
    setPlaybackSpeed(speed) {
        const wasBackwards = this.playbackSpeed < 0;
        const isNowBackwards = speed < 0;
        
        if (isNowBackwards) {
            // Negative speed: pause audio, backwards playhead movement
            this.playbackSpeed = speed;
            if (this.audioElement && this.isPlaying) {
                this.audioElement.pause(); // Pause audio completely
            }
            this.showMessage(`Backwards mode: ${Math.abs(speed)}x (silent)`);
        } else {
            // Positive speed: normal audio playback
            this.playbackSpeed = speed;
            if (this.audioElement) {
                // Always sync position and resume when switching to forward
                if (wasBackwards && !isNowBackwards && this.isPlaying) {
                    this.audioElement.currentTime = this.currentTime; // Sync audio to visual playhead
                    this.audioElement.playbackRate = speed;
                    
                    // Force resume playback from current position
                    this.audioElement.play().catch(e => {
                        console.log('Audio play failed:', e);
                    });
                } else {
                    this.audioElement.playbackRate = speed;
                }
            }
            this.showMessage(`Playback speed set to ${speed}x`);
        }
    }
    
    // Export functionality
    exportTiming() {
        if (this.selectedRegions.length === 0) {
            alert('No regions selected. Please select timing regions first.');
            return;
        }
        
        const timingData = this.selectedRegions
            .map(r => `${this.formatRegionValue(r.start)},${this.formatRegionValue(r.end)}`)
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