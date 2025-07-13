/**
 * Audio Wave Analyzer Regions Module
 * Handles drawing and management of timing regions (manual, auto-detected, grouped)
 */
export class AudioAnalyzerRegions {
    constructor(core) {
        this.core = core;
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
    
    drawAnalysisResults(ctx, width, height) {
        if (!this.core.waveformData || !this.core.waveformData.regions) return;
        
        const regions = this.core.waveformData.regions;
        
        // Draw detected regions with different colors
        regions.forEach(region => {
                const startX = this.core.timeToPixel(region.start);
                const endX = this.core.timeToPixel(region.end);
                
                if (endX >= 0 && startX <= width) {
                    // Get color based on region type
                    const color = this.getRegionColor(region);
                    
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
                    
                    // Draw region label with smart positioning
                    if (startX >= 0 && startX + 50 <= width) {
                        this.drawAutoRegionLabel(ctx, region, startX, width, height);
                    }
                }
            });
    }
    
    getRegionColor(region) {
        // Color based on region label/type
        let color = 'rgba(0, 255, 0, 0.2)'; // Default green
        
        // Check for grouped regions first - use original type for color
        if (region.metadata && region.metadata.type === 'grouped' && region.metadata.original_type) {
            // Use the original type to determine color for grouped regions
            const originalType = region.metadata.original_type;
            if (originalType === 'peak') {
                color = 'rgba(0, 150, 255, 0.25)'; // Blue/cyan for grouped peaks
            } else if (originalType === 'silence') {
                color = 'rgba(128, 128, 128, 0.3)'; // Gray for grouped silence
            } else if (originalType === 'word_boundary') {
                color = 'rgba(255, 255, 0, 0.2)'; // Yellow for grouped word boundaries
            } else if (originalType === 'speech') {
                color = 'rgba(34, 139, 34, 0.3)'; // Forest green for grouped speech
            }
        } else if (region.label === 'silence') {
            color = 'rgba(128, 128, 128, 0.3)'; // Gray for silence
        } else if (region.label === 'speech') {
            color = 'rgba(34, 139, 34, 0.3)'; // Forest green for speech (inverted silence)
        } else if (region.label.includes('word_boundary')) {
            color = 'rgba(255, 255, 0, 0.2)'; // Yellow for word boundaries
        } else if (region.label.includes('peak_')) {
            color = 'rgba(0, 150, 255, 0.25)'; // Blue/cyan for detected peaks
        } else if (region.label.includes('speech')) {
            color = 'rgba(0, 255, 0, 0.2)'; // Green for other speech types
        }
        
        return color;
    }
    
    // Smart labeling for auto-detected regions
    drawAutoRegionLabel(ctx, region, startX, width, height) {
        const isGrouped = region.metadata && region.metadata.type === 'grouped';
        
        // Apply density check to ALL auto-detected regions (both individual and grouped)
        if (!this.shouldShowAutoLabels(width)) {
            return; // Hide all auto-region labels when too dense - only show when zoomed in
        }
        
        // All auto regions use the same lower position (no stacking needed)
        const labelY = 30; // Lower than timeline numbers (which are at y=12)
        
        // Draw the label
        ctx.fillStyle = '#fff';
        ctx.font = '10px Arial';
        ctx.textAlign = 'left';
        
        const labelText = isGrouped ? 
            `${region.label}` : // Grouped: simpler label without confidence
            `${region.label} (${region.confidence.toFixed(2)})`; // Individual: with confidence
            
        ctx.fillText(labelText, startX + 2, labelY);
    }
    
    // Check if auto labels should be shown based on zoom/density
    shouldShowAutoLabels(width) {
        if (!this.core.waveformData || !this.core.waveformData.regions) return false;
        
        const regions = this.core.waveformData.regions;
        // Include ALL auto-detected regions (both individual and grouped) in density calculation
        const autoRegions = regions.filter(r => 
            !r.label.startsWith('Region') // Not manual (manual regions start with "Region")
        );
        
        // Calculate average pixels per region in visible area
        const visibleDuration = this.core.waveformData.duration / this.core.zoomLevel;
        const autoRegionsInView = autoRegions.filter(r => {
            const startTime = this.core.scrollOffset;
            const endTime = startTime + visibleDuration;
            return r.start < endTime && r.end > startTime;
        });
        
        if (autoRegionsInView.length === 0) return false;
        
        const pixelsPerRegion = width / autoRegionsInView.length;
        
        // Show labels only if there's enough space (at least 80px per region)
        return pixelsPerRegion >= 80;
    }
}