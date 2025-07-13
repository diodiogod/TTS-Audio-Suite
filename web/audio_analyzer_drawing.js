/**
 * Audio Wave Analyzer Drawing Module
 * Basic drawing utilities and helper functions for canvas rendering
 */
export class AudioAnalyzerDrawing {
    constructor(core) {
        this.core = core;
    }
    
    showInitialMessage(ctx, width, height) {
        ctx.fillStyle = this.core.colors.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Drop an audio file here or select one using the audio_file widget', width / 2, height / 2);
        
        ctx.font = '12px Arial';
        ctx.fillStyle = '#888';
        ctx.fillText('Supported formats: WAV, MP3, OGG, FLAC', width / 2, height / 2 + 20);
    }
    
    showFakeDataWarning(ctx, width, height) {
        // Draw warning background
        ctx.fillStyle = 'rgba(255, 165, 0, 0.8)';
        ctx.fillRect(0, 0, width, 40);
        
        // Draw warning text
        ctx.fillStyle = '#000';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('⚠️ FAKE TEST DATA - Connect proper audio source and analyze again', width / 2, 25);
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