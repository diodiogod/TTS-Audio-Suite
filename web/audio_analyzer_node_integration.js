/**
 * Audio Analyzer Node Integration Module
 * Handles communication with ComfyUI AudioAnalyzerNode
 */
export class AudioAnalyzerNodeIntegration {
    constructor(core) {
        this.core = core;
        this.lastAnalysisParams = null;
        this.isAnalyzing = false;
    }
    
    // Handle visualization data updates from node execution
    updateVisualization(data) {
        console.log('Received visualization data:', data);
        
        if (data.error) {
            this.core.showMessage(`Analysis error: ${data.error}`);
            this.core.ui.updateStatus('Analysis failed');
            return;
        }
        
        // Update waveform data - handle the correct data structure from Python
        this.core.waveformData = {
            samples: data.waveform?.samples || [],
            time: data.waveform?.time || [],
            rms: data.rms?.values || [],
            rmsTime: data.rms?.time || [],
            peaks: data.peaks || [],
            duration: data.duration || 0,
            sampleRate: data.sample_rate || 44100,
            regions: data.regions || [],
            analysisResults: data.analysis_results || {}
        };
        
        // Create audio element for playback if we have file path
        if (data.file_path) {
            this.setupAudioPlayback(data.file_path);
        }
        
        // Update UI status
        this.core.ui.updateStatus(`Analyzed: ${this.core.formatTime(this.core.waveformData.duration)}`);
        
        // Show analysis results summary
        this.showAnalysisResults(data.analysis_results);
        
        // Redraw visualization
        this.core.visualization.redraw();
        
        // Reset analysis state
        this.isAnalyzing = false;
        
        this.core.showMessage('Analysis complete');
    }
    
    // Handle audio file selection
    onAudioFileSelected(filePath) {
        console.log('Audio file selected:', filePath);
        
        if (!filePath || filePath.trim() === '') {
            this.core.showMessage('No audio file selected');
            return;
        }
        
        // Clear previous data
        this.core.waveformData = null;
        this.core.selectedRegions = [];
        this.core.clearSelection();
        
        // Update UI
        this.core.ui.updateStatus('Loading audio file...');
        this.core.visualization.redraw();
        
        // Trigger node execution
        this.triggerNodeExecution();
        
        this.core.showMessage(`Loading: ${filePath}`);
    }
    
    // Handle parameter changes
    onParametersChanged() {
        console.log('Analysis parameters changed');
        
        if (!this.hasAudioSource()) {
            this.core.showMessage('No audio source available for analysis');
            return;
        }
        
        // Check if parameters actually changed
        const currentParams = this.getCurrentAnalysisParams();
        if (this.paramsEqual(currentParams, this.lastAnalysisParams)) {
            console.log('Parameters unchanged, skipping re-analysis');
            return;
        }
        
        this.lastAnalysisParams = currentParams;
        
        // Clear previous analysis results
        if (this.core.waveformData) {
            this.core.waveformData.analysisResults = {};
        }
        
        // Update UI
        this.core.ui.updateStatus('Re-analyzing with new parameters...');
        this.core.visualization.redraw();
        
        // Trigger node execution
        this.triggerNodeExecution();
        
        this.core.showMessage('Re-analyzing audio...');
    }
    
    // Handle audio connection
    onAudioConnected() {
        console.log('Audio input connected');
        
        // Clear file-based data
        this.core.waveformData = null;
        this.core.selectedRegions = [];
        this.core.clearSelection();
        
        // Update UI
        this.core.ui.updateStatus('Audio connected, analyzing...');
        this.core.visualization.redraw();
        
        // Trigger node execution
        this.triggerNodeExecution();
        
        this.core.showMessage('Audio connected, analyzing...');
    }
    
    // Trigger node execution
    triggerNodeExecution() {
        if (this.isAnalyzing) {
            console.log('Analysis already in progress');
            return;
        }
        
        this.isAnalyzing = true;
        
        try {
            // Queue the node for execution
            if (this.core.node.graph && this.core.node.graph.runStep) {
                this.core.node.graph.runStep([this.core.node]);
            } else {
                // Fallback: manually trigger execution
                console.log('Triggering manual node execution');
                this.core.node.doExecute?.();
            }
        } catch (error) {
            console.error('Failed to trigger node execution:', error);
            this.core.showMessage(`Execution error: ${error.message}`);
            this.isAnalyzing = false;
        }
    }
    
    // Check if we have an audio source
    hasAudioSource() {
        // Check for file input
        const audioFileWidget = this.core.node.widgets?.find(w => w.name === 'audio_file');
        if (audioFileWidget && audioFileWidget.value && audioFileWidget.value.trim()) {
            return true;
        }
        
        // Check for audio input connection
        if (this.core.node.inputs && this.core.node.inputs.length > 1) {
            const audioInput = this.core.node.inputs[1]; // Assuming slot 1 is audio input
            if (audioInput && audioInput.link) {
                return true;
            }
        }
        
        return false;
    }
    
    // Get current analysis parameters
    getCurrentAnalysisParams() {
        const params = {};
        
        // Get widget values
        this.core.node.widgets?.forEach(widget => {
            if (widget.name === 'analysis_method' ||
                widget.name === 'silence_threshold' ||
                widget.name === 'silence_min_duration' ||
                widget.name === 'energy_sensitivity') {
                params[widget.name] = widget.value;
            }
        });
        
        return params;
    }
    
    // Compare parameter objects
    paramsEqual(params1, params2) {
        if (!params1 || !params2) return false;
        
        const keys1 = Object.keys(params1);
        const keys2 = Object.keys(params2);
        
        if (keys1.length !== keys2.length) return false;
        
        for (let key of keys1) {
            if (params1[key] !== params2[key]) return false;
        }
        
        return true;
    }
    
    // Setup audio playback
    setupAudioPlayback(filePath) {
        try {
            // Remove existing audio element
            if (this.core.audioElement) {
                this.core.audioElement.removeEventListener('ended', this.handleAudioEnded);
                this.core.audioElement.removeEventListener('timeupdate', this.handleTimeUpdate);
            }
            
            // Create new audio element
            this.core.audioElement = new Audio();
            this.core.audioElement.src = filePath;
            this.core.audioElement.preload = 'metadata';
            
            // Audio event listeners
            this.core.audioElement.addEventListener('ended', () => {
                this.core.isPlaying = false;
                this.core.ui.playButton.textContent = '▶️ Play';
                this.core.currentTime = 0;
                this.core.ui.updateTimeDisplay();
                this.core.visualization.redraw();
            });
            
            this.core.audioElement.addEventListener('timeupdate', () => {
                if (this.core.isPlaying) {
                    this.core.currentTime = this.core.audioElement.currentTime;
                    this.core.ui.updateTimeDisplay();
                }
            });
            
            this.core.audioElement.addEventListener('loadedmetadata', () => {
                console.log('Audio metadata loaded');
                this.core.ui.updateStatus('Audio loaded and ready for playback');
            });
            
            this.core.audioElement.addEventListener('error', (e) => {
                console.error('Audio loading error:', e);
                this.core.showMessage('Failed to load audio for playback');
            });
            
        } catch (error) {
            console.error('Failed to setup audio playback:', error);
            this.core.showMessage('Audio playback unavailable');
        }
    }
    
    // Show analysis results summary
    showAnalysisResults(results) {
        if (!results) return;
        
        const messages = [];
        
        if (results.silence_regions) {
            messages.push(`Found ${results.silence_regions.length} silence regions`);
        }
        
        if (results.speech_regions) {
            messages.push(`Found ${results.speech_regions.length} speech regions`);
        }
        
        if (results.energy_peaks) {
            messages.push(`Found ${results.energy_peaks.length} energy peaks`);
        }
        
        if (results.timing_markers) {
            messages.push(`Found ${results.timing_markers.length} timing markers`);
        }
        
        if (messages.length > 0) {
            this.core.showMessage(messages.join(', '));
        }
    }
    
    // Export analysis results
    exportAnalysisResults() {
        if (!this.core.waveformData || !this.core.waveformData.analysisResults) {
            this.core.showMessage('No analysis results to export');
            return null;
        }
        
        const results = {
            duration: this.core.waveformData.duration,
            sample_rate: this.core.waveformData.sampleRate,
            analysis_results: this.core.waveformData.analysisResults,
            selected_regions: this.core.selectedRegions,
            analysis_params: this.getCurrentAnalysisParams(),
            export_timestamp: new Date().toISOString()
        };
        
        return JSON.stringify(results, null, 2);
    }
    
    // Import analysis results
    importAnalysisResults(jsonData) {
        try {
            const results = JSON.parse(jsonData);
            
            if (results.analysis_results) {
                this.core.waveformData.analysisResults = results.analysis_results;
            }
            
            if (results.selected_regions) {
                this.core.selectedRegions = results.selected_regions;
                this.core.updateManualRegions();
            }
            
            this.core.visualization.redraw();
            this.core.showMessage('Analysis results imported successfully');
            
        } catch (error) {
            console.error('Failed to import analysis results:', error);
            this.core.showMessage('Failed to import analysis results');
        }
    }
    
    // Get node execution status
    getExecutionStatus() {
        return {
            isAnalyzing: this.isAnalyzing,
            hasAudioSource: this.hasAudioSource(),
            hasResults: !!(this.core.waveformData && this.core.waveformData.analysisResults),
            currentParams: this.getCurrentAnalysisParams()
        };
    }
}