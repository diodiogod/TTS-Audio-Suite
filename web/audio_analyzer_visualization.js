/**
 * Audio Wave Analyzer Visualization Module
 * Core waveform visualization and animation coordination
 */
import { AudioAnalyzerRegions } from './audio_analyzer_regions.js';
import { AudioAnalyzerDrawing } from './audio_analyzer_drawing.js';

export class AudioAnalyzerVisualization {
    constructor(core) {
        this.core = core;
        this.animationId = null;
        
        // Initialize sub-modules
        this.regions = new AudioAnalyzerRegions(core);
        this.drawing = new AudioAnalyzerDrawing(core);
        
        // Debug state
        this.dataStructureLogged = false;
        this.lastLogTime = null;
    }
    
    redraw() {
        if (!this.core.ctx) return;
        
        const canvas = this.core.canvas;
        const ctx = this.core.ctx;
        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;
        
        // Clear canvas
        ctx.fillStyle = this.core.colors.background;
        ctx.fillRect(0, 0, width, height);
        
        if (!this.core.waveformData) {
            this.drawing.showInitialMessage(ctx, width, height);
            return;
        }
        
        // Check for fake test data and show warning
        if (this.core.waveformData.duration === 10.79 && 
            this.core.waveformData.waveform && 
            this.core.waveformData.waveform.samples.length === 0) {
            this.drawing.showFakeDataWarning(ctx, width, height);
        }
        
        // Draw grid
        this.drawing.drawGrid(ctx, width, height);
        
        // Draw waveform
        this.drawWaveform(ctx, width, height);
        
        // Draw RMS if available
        if (this.core.waveformData.rms) {
            this.drawRMS(ctx, width, height);
        }
        
        // Draw selected regions
        this.regions.drawSelectedRegions(ctx, width, height);
        
        // Draw current selection
        this.drawing.drawCurrentSelection(ctx, width, height);
        
        // Draw loop markers
        this.drawing.drawLoopMarkers(ctx, width, height);
        
        // Draw playhead
        this.drawing.drawPlayhead(ctx, width, height);
        
        // Draw analysis results
        this.regions.drawAnalysisResults(ctx, width, height);
    }
    
    drawWaveform(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.samples) {
            console.warn('🎵 No waveform data available');
            return;
        }
        
        const samples = this.core.waveformData.samples;
        const duration = this.core.waveformData.duration;
        const sampleRate = this.core.waveformData.sampleRate;
        
        // Ensure we have valid data
        if (!samples.length || duration <= 0 || sampleRate <= 0) {
            console.warn('🎵 Invalid waveform data:', { samplesLength: samples.length, duration, sampleRate });
            return;
        }
        
        // Calculate visible time range
        const visibleDuration = duration / this.core.zoomLevel;
        const startTime = Math.max(0, this.core.scrollOffset);
        const endTime = Math.min(duration, startTime + visibleDuration);
        
        if (startTime >= endTime) {
            console.warn('🎵 Invalid time range:', { startTime, endTime });
            return;
        }
        
        // Set drawing style
        ctx.save(); // Save current state
        ctx.strokeStyle = this.core.colors.waveform; // Blue
        ctx.lineWidth = 2; // Make it visible
        ctx.globalAlpha = 1.0; // Ensure it's opaque
        
        // Draw the waveform
        ctx.beginPath();
        
        let pointsDrawn = 0;
        
        // Debug sample data structure first
        if (!this.dataStructureLogged) {
            // console.log('🎵 Sample data debug:', {  // Debug: sample data inspection
            //     samplesType: typeof samples,
            //     samplesLength: samples.length,
            //     isArray: Array.isArray(samples),
            //     firstSamples: samples.slice(0, 5),
            //     duration,
            //     sampleRate,
            //     calculatedSampleRate: samples.length / duration
            // });
            this.dataStructureLogged = true;
        }
        
        // Use the same approach as RMS - iterate through all pixels and check time bounds
        let hasStarted = false;
        
        for (let x = 0; x < width; x++) {
            const time = startTime + (x / width) * visibleDuration;
            
            // Only draw if the time is within audio duration (same check as RMS)
            if (time < 0 || time >= duration) {
                continue;
            }
            
            // Find corresponding sample index - use proper mapping
            const sampleIndex = Math.floor((time / duration) * samples.length);
            
            // Ensure we're within sample bounds
            if (sampleIndex < 0 || sampleIndex >= samples.length) {
                continue;
            }
            
            // Get sample value
            const sample = samples[sampleIndex];
            
            // Ensure we have a valid number
            if (typeof sample !== 'number' || isNaN(sample)) {
                console.warn(`🎵 Invalid sample at index ${sampleIndex}:`, sample);
                continue;
            }
            
            // Convert to screen coordinates (center line is height/2) using dynamic amplitude scaling
            const y = height/2 - (sample * height * this.core.amplitudeScale);
            
            // Draw line (same pattern as RMS)
            if (!hasStarted) {
                ctx.moveTo(x, y);
                hasStarted = true;
            } else {
                ctx.lineTo(x, y);
            }
            pointsDrawn++;
        }
        
        // Actually draw the path
        ctx.stroke();
        ctx.restore(); // Restore previous state
        
            // Minimal debug logging (only once every 3 seconds) - but also check if animation should be running
        if (!this.lastLogTime || Date.now() - this.lastLogTime > 3000) {
            // console.log(`🎵 Waveform drawn: ${samples.length} samples, ${duration.toFixed(2)}s, zoom=${this.core.zoomLevel.toFixed(2)}, points=${pointsDrawn}, isPlaying=${this.core.isPlaying}, animationId=${this.animationId}`);  // Debug: render stats
            this.lastLogTime = Date.now();
        }
    }
    
    drawRMS(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.rms) return;
        
        const rmsData = this.core.waveformData.rms;
        
        // Handle both old and new RMS data structures
        let rmsValues, rmsTime;
        if (rmsData.values && rmsData.time) {
            // New structure with separate values and time arrays
            rmsValues = rmsData.values;
            rmsTime = rmsData.time;
        } else if (Array.isArray(rmsData)) {
            // Old structure - array of values only
            rmsValues = rmsData;
            rmsTime = null;
        } else {
            return; // Invalid structure
        }
        
        // Safety check
        if (!rmsValues || rmsValues.length === 0) return;
        
        const duration = this.core.waveformData.duration;
        const visibleDuration = duration / this.core.zoomLevel;
        const startTime = this.core.scrollOffset;
        const endTime = startTime + visibleDuration;
        
        ctx.strokeStyle = this.core.colors.rms;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        let hasStarted = false;
        
        for (let x = 0; x < width; x++) {
            const time = startTime + (x / width) * visibleDuration;
            let rmsValue;
            
            if (rmsTime && rmsTime.length > 0) {
                // Use time-based indexing (new structure)
                let closestIndex = 0;
                let minDistance = Math.abs(rmsTime[0] - time);
                
                for (let i = 1; i < rmsTime.length; i++) {
                    const distance = Math.abs(rmsTime[i] - time);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestIndex = i;
                    } else {
                        break; // Since times are sorted, we can stop
                    }
                }
                
                if (closestIndex < rmsValues.length) {
                    rmsValue = rmsValues[closestIndex];
                } else {
                    continue;
                }
            } else {
                // Use uniform distribution (old structure)
                const rmsIndex = Math.floor((time / duration) * rmsValues.length);
                if (rmsIndex >= 0 && rmsIndex < rmsValues.length) {
                    rmsValue = rmsValues[rmsIndex];
                } else {
                    continue;
                }
            }
            
            // Only draw if the time is within visible range
            if (time >= startTime && time <= endTime) {
                const y = height/2 - (rmsValue * height * this.core.amplitudeScale);
                
                if (!hasStarted) {
                    ctx.moveTo(x, y);
                    hasStarted = true;
                } else {
                    ctx.lineTo(x, y);
                }
            }
        }
        
        ctx.stroke();
    }
    
    // Animation helpers
    startAnimation() {
        const animate = () => {
            this.redraw();
            if (this.core.isPlaying) {
                this.animationId = requestAnimationFrame(animate);
            }
        };
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.animationId = requestAnimationFrame(animate);
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    // Delegate methods to sub-modules for backward compatibility
    getColorForFrequency(frequency) {
        return this.drawing.getColorForFrequency(frequency);
    }
    
    getIntensityAlpha(intensity) {
        return this.drawing.getIntensityAlpha(intensity);
    }
    
    exportAsImage() {
        return this.drawing.exportAsImage();
    }
}