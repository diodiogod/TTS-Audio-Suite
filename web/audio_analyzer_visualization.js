/**
 * Audio Analyzer Visualization Module
 * All canvas drawing and visualization functionality
 */
export class AudioAnalyzerVisualization {
    constructor(core) {
        this.core = core;
        this.animationId = null;
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
            this.showInitialMessage();
            return;
        }
        
        // Draw grid
        this.drawGrid(ctx, width, height);
        
        // Draw waveform
        this.drawWaveform(ctx, width, height);
        
        // Draw RMS if available
        if (this.core.waveformData.rms) {
            this.drawRMS(ctx, width, height);
        }
        
        // Draw selected regions
        this.drawSelectedRegions(ctx, width, height);
        
        // Draw current selection
        this.drawCurrentSelection(ctx, width, height);
        
        // Draw playhead
        this.drawPlayhead(ctx, width, height);
        
        // Draw analysis results
        this.drawAnalysisResults(ctx, width, height);
    }
    
    showInitialMessage() {
        const canvas = this.core.canvas;
        const ctx = this.core.ctx;
        const width = canvas.width / devicePixelRatio;
        const height = canvas.height / devicePixelRatio;
        
        ctx.fillStyle = this.core.colors.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Drop an audio file here or select one using the audio_file widget', width / 2, height / 2);
        
        ctx.font = '12px Arial';
        ctx.fillStyle = '#888';
        ctx.fillText('Supported formats: WAV, MP3, OGG, FLAC', width / 2, height / 2 + 20);
    }
    
    drawGrid(ctx, width, height) {
        if (!this.core.waveformData) return;
        
        ctx.strokeStyle = this.core.colors.grid;
        ctx.lineWidth = 1;
        
        // Time grid
        const visibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
        const startTime = this.core.scrollOffset;
        const endTime = startTime + visibleDuration;
        
        // Calculate appropriate time interval
        let timeInterval = 1; // seconds
        if (visibleDuration > 300) timeInterval = 60;
        else if (visibleDuration > 60) timeInterval = 10;
        else if (visibleDuration > 10) timeInterval = 2;
        else if (visibleDuration > 2) timeInterval = 0.5;
        else if (visibleDuration > 0.5) timeInterval = 0.1;
        else timeInterval = 0.05;
        
        // Draw vertical time lines
        const firstLine = Math.ceil(startTime / timeInterval) * timeInterval;
        for (let time = firstLine; time <= endTime; time += timeInterval) {
            const x = this.core.timeToPixel(time);
            if (x >= 0 && x <= width) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
                
                // Draw time labels
                ctx.fillStyle = this.core.colors.text;
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(this.core.formatTime(time), x, 12);
            }
        }
        
        // Draw horizontal amplitude lines
        const amplitudeLines = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8];
        amplitudeLines.forEach(amp => {
            const y = height/2 - (amp * height * 0.4);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
            
            // Draw amplitude labels
            if (amp !== 0) {
                ctx.fillStyle = this.core.colors.text;
                ctx.font = '9px Arial';
                ctx.textAlign = 'right';
                ctx.fillText(amp.toFixed(1), width - 5, y - 2);
            }
        });
    }
    
    drawWaveform(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.samples) return;
        
        const samples = this.core.waveformData.samples;
        const duration = this.core.waveformData.duration;
        const sampleRate = this.core.waveformData.sampleRate;
        const visibleDuration = duration / this.core.zoomLevel;
        const startTime = this.core.scrollOffset;
        const endTime = startTime + visibleDuration;
        
        // Calculate sample range
        const startSample = Math.max(0, Math.floor(startTime * sampleRate));
        const endSample = Math.min(samples.length, Math.ceil(endTime * sampleRate));
        
        if (startSample >= endSample) return;
        
        // Calculate samples per pixel
        const samplesPerPixel = Math.max(1, Math.floor((endSample - startSample) / width));
        
        ctx.strokeStyle = this.core.colors.waveform;
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        let hasStarted = false;
        
        for (let x = 0; x < width; x++) {
            const sampleIndex = startSample + Math.floor((x / width) * (endSample - startSample));
            
            if (sampleIndex >= samples.length) break;
            
            let min = 0, max = 0;
            
            // Get min/max for this pixel
            for (let i = 0; i < samplesPerPixel && sampleIndex + i < samples.length; i++) {
                const sample = samples[sampleIndex + i];
                min = Math.min(min, sample);
                max = Math.max(max, sample);
            }
            
            // Convert to canvas coordinates
            const y1 = height/2 - (min * height * 0.4);
            const y2 = height/2 - (max * height * 0.4);
            
            if (!hasStarted) {
                ctx.moveTo(x, y1);
                hasStarted = true;
            }
            
            // Draw vertical line for this pixel
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
        }
        
        ctx.stroke();
    }
    
    drawRMS(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.rms) return;
        
        const rms = this.core.waveformData.rms;
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
            const rmsIndex = Math.floor((time / duration) * rms.length);
            
            if (rmsIndex >= 0 && rmsIndex < rms.length) {
                const rmsValue = rms[rmsIndex];
                const y = height/2 - (rmsValue * height * 0.4);
                
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
    
    drawSelectedRegions(ctx, width, height) {
        if (!this.core.selectedRegions || this.core.selectedRegions.length === 0) return;
        
        this.core.selectedRegions.forEach((region, index) => {
            const startX = this.core.timeToPixel(region.start);
            const endX = this.core.timeToPixel(region.end);
            
            if (endX >= 0 && startX <= width) {
                // Draw region background
                ctx.fillStyle = this.core.colors.region;
                ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
                
                // Draw region borders
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 2;
                ctx.beginPath();
                if (startX >= 0 && startX <= width) {
                    ctx.moveTo(startX, 0);
                    ctx.lineTo(startX, height);
                }
                if (endX >= 0 && endX <= width) {
                    ctx.moveTo(endX, 0);
                    ctx.lineTo(endX, height);
                }
                ctx.stroke();
                
                // Draw region label
                const labelX = Math.max(5, Math.min(width - 50, startX + 5));
                ctx.fillStyle = '#00ff00';
                ctx.font = '11px Arial';
                ctx.textAlign = 'left';
                ctx.fillText(region.label, labelX, 20 + (index * 15));
            }
        });
    }
    
    drawCurrentSelection(ctx, width, height) {
        if (this.core.selectedStart === null || this.core.selectedEnd === null) return;
        
        const startX = this.core.timeToPixel(this.core.selectedStart);
        const endX = this.core.timeToPixel(this.core.selectedEnd);
        
        if (endX >= 0 && startX <= width) {
            // Draw selection background
            ctx.fillStyle = this.core.colors.selection;
            ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
            
            // Draw selection borders
            ctx.strokeStyle = '#ffff00';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            if (startX >= 0 && startX <= width) {
                ctx.moveTo(startX, 0);
                ctx.lineTo(startX, height);
            }
            if (endX >= 0 && endX <= width) {
                ctx.moveTo(endX, 0);
                ctx.lineTo(endX, height);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
    
    drawPlayhead(ctx, width, height) {
        if (!this.core.waveformData) return;
        
        const playheadX = this.core.timeToPixel(this.core.currentTime);
        
        if (playheadX >= 0 && playheadX <= width) {
            ctx.strokeStyle = this.core.colors.playhead;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, height);
            ctx.stroke();
            
            // Draw playhead indicator
            ctx.fillStyle = this.core.colors.playhead;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX - 5, 10);
            ctx.lineTo(playheadX + 5, 10);
            ctx.closePath();
            ctx.fill();
        }
    }
    
    drawAnalysisResults(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.analysisResults) return;
        
        const results = this.core.waveformData.analysisResults;
        
        // Draw silence regions
        if (results.silence_regions) {
            results.silence_regions.forEach(region => {
                const startX = this.core.timeToPixel(region.start);
                const endX = this.core.timeToPixel(region.end);
                
                if (endX >= 0 && startX <= width) {
                    ctx.fillStyle = 'rgba(128, 128, 128, 0.3)';
                    ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
                }
            });
        }
        
        // Draw speech regions
        if (results.speech_regions) {
            results.speech_regions.forEach(region => {
                const startX = this.core.timeToPixel(region.start);
                const endX = this.core.timeToPixel(region.end);
                
                if (endX >= 0 && startX <= width) {
                    ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(Math.max(0, startX), height - 10);
                    ctx.lineTo(Math.min(width, endX), height - 10);
                    ctx.stroke();
                }
            });
        }
        
        // Draw energy peaks
        if (results.energy_peaks) {
            results.energy_peaks.forEach(peak => {
                const peakX = this.core.timeToPixel(peak.time);
                
                if (peakX >= 0 && peakX <= width) {
                    ctx.fillStyle = 'rgba(255, 165, 0, 0.8)';
                    ctx.beginPath();
                    ctx.arc(peakX, height/2, 3, 0, Math.PI * 2);
                    ctx.fill();
                }
            });
        }
        
        // Draw timing markers
        if (results.timing_markers) {
            results.timing_markers.forEach(marker => {
                const markerX = this.core.timeToPixel(marker.time);
                
                if (markerX >= 0 && markerX <= width) {
                    ctx.strokeStyle = 'rgba(255, 0, 255, 0.9)';
                    ctx.lineWidth = 2;
                    ctx.setLineDash([3, 3]);
                    ctx.beginPath();
                    ctx.moveTo(markerX, 0);
                    ctx.lineTo(markerX, height);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    
                    // Draw marker label
                    ctx.fillStyle = 'rgba(255, 0, 255, 0.9)';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(marker.label || 'Marker', markerX, height - 5);
                }
            });
        }
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
    
    // Utility methods
    getColorForFrequency(frequency) {
        // Color-code frequency ranges
        if (frequency < 200) return '#ff4444'; // Low frequencies - red
        if (frequency < 1000) return '#ffaa44'; // Mid-low frequencies - orange
        if (frequency < 4000) return '#44ff44'; // Mid frequencies - green
        if (frequency < 8000) return '#44aaff'; // Mid-high frequencies - blue
        return '#aa44ff'; // High frequencies - purple
    }
    
    getIntensityAlpha(intensity) {
        // Convert intensity to alpha value (0-1)
        return Math.max(0.1, Math.min(1, intensity / 100));
    }
    
    // Export visualization as image
    exportAsImage() {
        if (!this.core.canvas) return null;
        
        try {
            return this.core.canvas.toDataURL('image/png');
        } catch (e) {
            console.error('Failed to export visualization:', e);
            return null;
        }
    }
}