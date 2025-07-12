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
        
        // Draw loop markers
        this.drawLoopMarkers(ctx, width, height);
        
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
        
        // Draw horizontal amplitude lines with dynamic scaling
        const maxAmp = this.core.maxAmplitudeRange;
        const amplitudeLines = [-maxAmp * 0.8, -maxAmp * 0.6, -maxAmp * 0.4, -maxAmp * 0.2, 0, maxAmp * 0.2, maxAmp * 0.4, maxAmp * 0.6, maxAmp * 0.8];
        amplitudeLines.forEach(amp => {
            const y = height/2 - (amp * height * this.core.amplitudeScale);
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
    
    drawSelectedRegions(ctx, width, height) {
        if (!this.core.selectedRegions || this.core.selectedRegions.length === 0) return;
        
        this.core.selectedRegions.forEach((region, index) => {
            const startX = this.core.timeToPixel(region.start);
            const endX = this.core.timeToPixel(region.end);
            
            if (endX >= 0 && startX <= width) {
                // Choose color based on state
                let fillColor = this.core.colors.region;
                let strokeColor = '#00ff00';
                let lineWidth = 2;
                
                if (this.core.selectedRegionIndices.includes(index)) {
                    // Selected for deletion (multiple selection)
                    fillColor = this.core.colors.regionSelected;
                    strokeColor = '#ff8c00';
                    lineWidth = 3;
                } else if (index === this.core.highlightedRegionIndex) {
                    // Highlighted (click-to-highlight, persists)
                    fillColor = this.core.colors.regionHovered;
                    strokeColor = '#00ff00';
                    lineWidth = 2;
                }
                
                // Draw region background
                ctx.fillStyle = fillColor;
                ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
                
                // Draw region borders
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth;
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
                
                // Draw region label with number
                const labelX = Math.max(5, Math.min(width - 80, startX + 5));
                ctx.fillStyle = strokeColor;
                ctx.font = this.core.selectedRegionIndices.includes(index) ? 'bold 12px Arial' : '11px Arial';
                ctx.textAlign = 'left';
                const labelText = `${index + 1}. ${region.label}`;
                ctx.fillText(labelText, labelX, 20 + (index * 15));
                
                // Show deletion hint for selected regions
                if (this.core.selectedRegionIndices.includes(index)) {
                    ctx.fillStyle = '#ff8c00';
                    ctx.font = '10px Arial';
                    const hint = this.core.selectedRegionIndices.length > 1 ? 
                        `(${this.core.selectedRegionIndices.length} selected - Delete to remove)` :
                        '(Press Delete to remove)';
                    ctx.fillText(hint, labelX, 35 + (index * 15));
                }
            }
        });
    }
    
    drawLoopMarkers(ctx, width, height) {
        if (this.core.loopStart === null || this.core.loopEnd === null) return;
        
        const startX = this.core.timeToPixel(this.core.loopStart);
        const endX = this.core.timeToPixel(this.core.loopEnd);
        const markerHeight = 20;
        const markerY = height - markerHeight;
        
        // Draw subtle loop region background
        if (endX >= 0 && startX <= width) {
            ctx.fillStyle = 'rgba(255, 0, 255, 0.1)';
            ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
        }
        
        // Draw loop start marker (triangle pointing right)
        if (startX >= 0 && startX <= width) {
            ctx.fillStyle = this.core.colors.loopMarker;
            ctx.beginPath();
            ctx.moveTo(startX, height);
            ctx.lineTo(startX - 8, markerY);
            ctx.lineTo(startX + 8, markerY);
            ctx.closePath();
            ctx.fill();
            
            // Draw start label
            ctx.fillStyle = this.core.colors.loopMarker;
            ctx.font = 'bold 10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('START', startX, markerY - 5);
        }
        
        // Draw loop end marker (triangle pointing left)
        if (endX >= 0 && endX <= width) {
            ctx.fillStyle = this.core.colors.loopMarker;
            ctx.beginPath();
            ctx.moveTo(endX, height);
            ctx.lineTo(endX - 8, markerY);
            ctx.lineTo(endX + 8, markerY);
            ctx.closePath();
            ctx.fill();
            
            // Draw end label
            ctx.fillStyle = this.core.colors.loopMarker;
            ctx.font = 'bold 10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('END', endX, markerY - 5);
        }
        
        // Draw loop indicator in corner
        if (this.core.isLooping) {
            ctx.fillStyle = this.core.colors.loopMarker;
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'right';
            ctx.fillText('🔄 LOOPING', width - 10, height - 10);
        }
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
        if (!this.core.waveformData || !this.core.waveformData.regions) return;
        
        const regions = this.core.waveformData.regions;
        
        // Draw detected regions with different colors
        regions.forEach(region => {
                const startX = this.core.timeToPixel(region.start);
                const endX = this.core.timeToPixel(region.end);
                
                if (endX >= 0 && startX <= width) {
                    // Color based on region label/type
                    let color = 'rgba(0, 255, 0, 0.2)'; // Default green
                    
                    if (region.label === 'silence') {
                        color = 'rgba(128, 128, 128, 0.3)'; // Gray for silence
                    } else if (region.label.includes('word_boundary')) {
                        color = 'rgba(255, 255, 0, 0.2)'; // Yellow for word boundaries
                    } else if (region.label.includes('peak_')) {
                        color = 'rgba(0, 150, 255, 0.25)'; // Blue/cyan for detected peaks
                    } else if (region.label.includes('speech')) {
                        color = 'rgba(0, 255, 0, 0.2)'; // Green for speech
                    }
                    
                    // Draw region background
                    ctx.fillStyle = color;
                    ctx.fillRect(Math.max(0, startX), 0, Math.min(width, endX) - Math.max(0, startX), height);
                    
                    // Draw region border
                    ctx.strokeStyle = color.replace('0.2', '0.8'); // More opaque border
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(startX, 0);
                    ctx.lineTo(startX, height);
                    ctx.moveTo(endX, 0);
                    ctx.lineTo(endX, height);
                    ctx.stroke();
                    
                    // Draw region label
                    if (startX >= 0 && startX + 50 <= width) {
                        ctx.fillStyle = '#fff';
                        ctx.font = '10px Arial';
                        ctx.textAlign = 'left';
                        ctx.fillText(`${region.label} (${region.confidence.toFixed(2)})`, startX + 2, 15);
                    }
                }
            });
        
        // Energy peaks would be drawn here if available
        // Region analysis complete
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