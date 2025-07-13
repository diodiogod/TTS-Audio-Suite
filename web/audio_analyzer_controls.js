/**
 * Audio Wave Analyzer Controls Module
 * Handles creation and management of all UI controls
 */
export class AudioAnalyzerControls {
    constructor(core) {
        this.core = core;
    }
    
    createPlaybackControls() {
        const playbackContainer = document.createElement('div');
        playbackContainer.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Play button
        this.core.ui.playButton = document.createElement('button');
        this.core.ui.playButton.textContent = '▶️ Play';
        this.core.ui.playButton.style.cssText = `
            padding: 4px 8px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.core.ui.playButton.onclick = () => this.core.togglePlayback();
        
        // Stop button
        this.core.ui.stopButton = document.createElement('button');
        this.core.ui.stopButton.textContent = '⏹️ Stop';
        this.core.ui.stopButton.style.cssText = `
            padding: 4px 8px;
            background: #666;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.core.ui.stopButton.onclick = () => this.core.stopPlayback();
        
        // Time display
        this.core.ui.timeDisplay = document.createElement('span');
        this.core.ui.timeDisplay.textContent = '00:00.000';
        this.core.ui.timeDisplay.style.cssText = `
            font-family: monospace;
            color: #fff;
            font-size: 11px;
            margin-left: 8px;
        `;
        
        
        // Selection display
        this.core.ui.selectionDisplay = document.createElement('span');
        this.core.ui.selectionDisplay.textContent = 'No selection';
        this.core.ui.selectionDisplay.style.cssText = `
            font-family: monospace;
            color: #ffff00;
            font-size: 11px;
            margin-left: 16px;
        `;
        
        playbackContainer.appendChild(this.core.ui.playButton);
        playbackContainer.appendChild(this.core.ui.stopButton);
        playbackContainer.appendChild(this.core.ui.timeDisplay);
        playbackContainer.appendChild(this.core.ui.selectionDisplay);
        
        return playbackContainer;
    }
    
    createSpeedSlider() {
        const sliderContainer = document.createElement('div');
        sliderContainer.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 4px 0;
            gap: 8px;
        `;
        
        const speedLabel = document.createElement('span');
        speedLabel.textContent = 'Speed:';
        speedLabel.style.cssText = `
            color: #fff;
            font-size: 10px;
            font-weight: bold;
        `;
        
        this.core.ui.speedSlider = document.createElement('input');
        this.core.ui.speedSlider.type = 'range';
        this.core.ui.speedSlider.min = '0.25';
        this.core.ui.speedSlider.max = '2';
        this.core.ui.speedSlider.step = '0.05';
        this.core.ui.speedSlider.value = '1';
        this.core.ui.speedSlider.style.cssText = `
            width: 120px;
            height: 4px;
            background: #333;
            outline: none;
            border-radius: 2px;
            cursor: pointer;
        `;
        
        // Style the slider thumb (webkit browsers)
        const style = document.createElement('style');
        style.textContent = `
            input[type="range"]::-webkit-slider-thumb {
                appearance: none;
                width: 12px;
                height: 12px;
                background: white;
                border-radius: 50%;
                cursor: pointer;
            }
            input[type="range"]::-moz-range-thumb {
                width: 12px;
                height: 12px;
                background: white;
                border-radius: 50%;
                cursor: pointer;
                border: none;
            }
        `;
        document.head.appendChild(style);
        
        this.core.ui.speedValue = document.createElement('span');
        this.core.ui.speedValue.textContent = '1x';
        this.core.ui.speedValue.style.cssText = `
            color: #fff;
            font-size: 10px;
            font-weight: bold;
            min-width: 24px;
            text-align: center;
        `;
        
        this.core.ui.speedSlider.oninput = () => {
            const speed = parseFloat(this.core.ui.speedSlider.value);
            this.core.ui.speedValue.textContent = `${speed.toFixed(2)}x`;
            this.core.setPlaybackSpeed(speed);
        };
        
        sliderContainer.appendChild(speedLabel);
        sliderContainer.appendChild(this.core.ui.speedSlider);
        sliderContainer.appendChild(this.core.ui.speedValue);
        
        return sliderContainer;
    }
    
    createMainControls() {
        // Single row with all main action buttons
        const mainControls = document.createElement('div');
        mainControls.style.cssText = `
            display: flex;
            gap: 6px;
            align-items: center;
            padding: 2px 0;
            flex-wrap: wrap;
            justify-content: space-between;
        `;
        
        const regionGroup = document.createElement('div');
        regionGroup.style.cssText = 'display: flex; gap: 6px; align-items: center;';

        const loopGroup = document.createElement('div');
        loopGroup.style.cssText = 'display: flex; gap: 6px; align-items: center;';

        const buttonStyle = `
            padding: 3px 6px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            color: white;
        `;
        
        // Upload button
        const uploadButton = document.createElement('button');
        uploadButton.textContent = '📁 Upload Audio';
        uploadButton.style.cssText = buttonStyle + 'background: #007bff; font-weight: bold;';
        uploadButton.onclick = () => this.handleUploadClick();
        
        
        // Delete region button
        const deleteButton = document.createElement('button');
        deleteButton.textContent = '🗑️ Delete Region';
        deleteButton.style.cssText = buttonStyle + 'background: #d35400;';
        deleteButton.onclick = () => {
            if (this.core.selectedRegionIndices.length > 0 || this.core.highlightedRegionIndex >= 0) {
                this.core.deleteSelectedRegion();
            } else {
                this.core.showMessage('No region selected. Click a region to highlight it for deletion.');
            }
        };
        
        // Add region button
        const addRegionButton = document.createElement('button');
        addRegionButton.textContent = '➕ Add Region';
        addRegionButton.style.cssText = buttonStyle + 'background: #17a2b8;';
        addRegionButton.onclick = () => this.core.addSelectedRegion();
        
        // Clear all regions button
        const clearAllButton = document.createElement('button');
        clearAllButton.textContent = '🗑️ Clear All';
        clearAllButton.style.cssText = buttonStyle + 'background: #803300ff;';
        clearAllButton.onclick = () => this.core.clearAllRegions();
        
        // Set loop button
        const setLoopButton = document.createElement('button');
        setLoopButton.textContent = '🔻 Set Loop';
        setLoopButton.style.cssText = buttonStyle + 'background: #7140cdff;';
        setLoopButton.onclick = () => this.core.setLoopFromSelection();
        
        // Toggle looping button
        const toggleLoopButton = document.createElement('button');
        toggleLoopButton.textContent = '🔄 Loop ON/OFF';
        toggleLoopButton.style.cssText = buttonStyle + 'background: #583d8dff;';
        toggleLoopButton.onclick = () => this.core.toggleLooping();
        
        // Clear loop button
        const clearLoopButton = document.createElement('button');
        clearLoopButton.textContent = '🚫 Clear Loop';
        clearLoopButton.style.cssText = buttonStyle + 'background: #2f204bff;'; 
        clearLoopButton.onclick = () => this.core.clearLoopMarkers();
        
        // Export timing button
        const exportButton = document.createElement('button');
        exportButton.textContent = '📋 Export Timings';
        exportButton.style.cssText = buttonStyle + 'background: #6c757d;'; 
        exportButton.onclick = () => this.core.exportTiming();
        
        regionGroup.appendChild(uploadButton);
        regionGroup.appendChild(addRegionButton);
        regionGroup.appendChild(deleteButton);
        regionGroup.appendChild(clearAllButton);

        loopGroup.appendChild(setLoopButton);
        loopGroup.appendChild(toggleLoopButton);
        loopGroup.appendChild(clearLoopButton);

        mainControls.appendChild(regionGroup);
        mainControls.appendChild(loopGroup);
        mainControls.appendChild(exportButton);
        
        return mainControls;
    }
    
    createZoomControls() {
        const zoomControls = document.createElement('div');
        zoomControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
            margin-left: auto;
        `;
        
        // Zoom in button
        const zoomInButton = document.createElement('button');
        zoomInButton.textContent = '🔍+';
        zoomInButton.style.cssText = `
            padding: 4px 8px;
            background: #46255a; /* Muted purple-gray to de-emphasize */
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomInButton.onclick = () => this.core.zoomIn();
        
        // Zoom out button
        const zoomOutButton = document.createElement('button');
        zoomOutButton.textContent = '🔍-';
        zoomOutButton.style.cssText = `
            padding: 4px 8px;
            background: #46255a; /* Muted purple-gray to de-emphasize */
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomOutButton.onclick = () => this.core.zoomOut();
        
        // Reset zoom button
        const resetZoomButton = document.createElement('button');
        resetZoomButton.textContent = '🔄 Reset';
        resetZoomButton.style.cssText = `
            padding: 4px 8px;
            background: #46255a; /* Muted purple-gray to de-emphasize */
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        resetZoomButton.onclick = () => this.core.resetZoom();
        
        zoomControls.appendChild(zoomInButton);
        zoomControls.appendChild(zoomOutButton);
        zoomControls.appendChild(resetZoomButton);
        
        return zoomControls;
    }
    
    createStatusDisplays() {
        const statusContainer = document.createElement('div');
        
        // Status display
        this.core.ui.statusDisplay = document.createElement('div');
        this.core.ui.statusDisplay.style.cssText = `
            font-size: 11px;
            color: #888;
            padding: 2px 0;
            border-top: 1px solid #333;
            margin-top: 4px;
        `;
        this.core.ui.statusDisplay.textContent = 'Ready to analyze audio';
        
        // Message display
        this.core.ui.messageDisplay = document.createElement('div');
        this.core.ui.messageDisplay.style.cssText = `
            font-size: 11px;
            color: #4a9eff;
            padding: 2px 0;
            min-height: 14px;
        `;
        
        statusContainer.appendChild(this.core.ui.statusDisplay);
        statusContainer.appendChild(this.core.ui.messageDisplay);
        
        return statusContainer;
    }
    
    setupCanvasSize() {
        const rect = this.core.canvas.getBoundingClientRect();
        this.core.canvas.width = rect.width * devicePixelRatio;
        this.core.canvas.height = rect.height * devicePixelRatio;
        this.core.ctx.scale(devicePixelRatio, devicePixelRatio);
    }
    
    setupCanvasResize() {
        const resizeObserver = new ResizeObserver(() => {
            this.core.resizeCanvas();
        });
        resizeObserver.observe(this.core.canvas);
    }
    
    setupDragAndDrop() {
        // Add drag and drop functionality
        this.core.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '0.7';
        });
        
        this.core.canvas.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '1';
        });
        
        this.core.canvas.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '1';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('audio/')) {
                    try {
                        this.showMessage('Uploading audio file...');
                        this.updateStatus('Uploading...');
                        
                        // Upload file to ComfyUI
                        const uploadResult = await this.uploadFileToComfyUI(file);
                        
                        if (uploadResult.success) {
                            // Update the audio_file widget with the full path to the uploaded file
                            const audioFileWidget = this.core.node.widgets.find(w => w.name === 'audio_file');
                            if (audioFileWidget) {
                                // Construct the full path to the uploaded file in ComfyUI's input directory
                                const fullPath = uploadResult.subfolder ? 
                                    `${uploadResult.subfolder}/${uploadResult.filename}` : 
                                    uploadResult.filename;
                                
                                audioFileWidget.value = fullPath;
                                
                                // Trigger widget callback to update the display
                                if (audioFileWidget.callback) {
                                    audioFileWidget.callback(fullPath);
                                }
                            }
                            
                            this.showMessage(`File uploaded: ${uploadResult.filename}. Click Analyze to process.`);
                            this.updateStatus('File uploaded - ready to analyze');
                        } else {
                            this.showMessage(`Upload failed: ${uploadResult.error || 'Unknown error'}`);
                            this.updateStatus('Upload failed');
                        }
                    } catch (error) {
                        console.error('Drag & drop upload error:', error);
                        this.showMessage(`Upload failed: ${error.message}`);
                        this.updateStatus('Upload failed');
                    }
                } else {
                    this.showMessage('Please drop an audio file');
                }
            }
        });
    }
    
    // Handle upload button click
    handleUploadClick() {
        // Create file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'audio/*';
        fileInput.style.display = 'none';
        
        fileInput.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            
            try {
                this.showMessage('Uploading audio file...');
                this.updateStatus('Uploading...');
                
                // Upload file to ComfyUI
                const uploadResult = await this.uploadFileToComfyUI(file);
                
                if (uploadResult.success) {
                    // Update the audio_file widget with the full path to the uploaded file
                    const audioFileWidget = this.core.node.widgets.find(w => w.name === 'audio_file');
                    if (audioFileWidget) {
                        // Construct the full path to the uploaded file in ComfyUI's input directory
                        const fullPath = uploadResult.subfolder ? 
                            `${uploadResult.subfolder}/${uploadResult.filename}` : 
                            uploadResult.filename;
                        
                        audioFileWidget.value = fullPath;
                        
                        // Trigger widget callback to update the display
                        if (audioFileWidget.callback) {
                            audioFileWidget.callback(fullPath);
                        }
                    }
                    
                    this.showMessage(`File uploaded: ${uploadResult.filename}. Click Analyze to process.`);
                    this.updateStatus('File uploaded - ready to analyze');
                } else {
                    this.showMessage(`Upload failed: ${uploadResult.error || 'Unknown error'}`);
                    this.updateStatus('Upload failed');
                }
            } catch (error) {
                console.error('Upload button error:', error);
                this.showMessage(`Upload failed: ${error.message}`);
                this.updateStatus('Upload failed');
            }
            
            // Clean up
            document.body.removeChild(fileInput);
        };
        
        // Trigger file picker
        document.body.appendChild(fileInput);
        fileInput.click();
    }
    
    // Upload file to ComfyUI input directory
    async uploadFileToComfyUI(file) {
        try {
            const formData = new FormData();
            formData.append('image', file); // ComfyUI expects 'image' parameter even for audio
            formData.append('type', 'input');
            formData.append('subfolder', '');
            
            const response = await fetch('/upload/image', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            
            return {
                success: true,
                filename: result.name,
                subfolder: result.subfolder || ''
            };
        } catch (error) {
            console.error('File upload failed:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    // Update time display
    updateTimeDisplay() {
        if (this.core.ui.timeDisplay) {
            this.core.ui.timeDisplay.textContent = this.core.formatTime(this.core.currentTime);
        }
    }
    
    // Update selection display
    updateSelectionDisplay() {
        if (this.core.ui.selectionDisplay) {
            if (this.core.selectedStart !== null && this.core.selectedEnd !== null) {
                const duration = this.core.selectedEnd - this.core.selectedStart;
                this.core.ui.selectionDisplay.textContent = 
                    `Selected: ${this.core.formatTime(this.core.selectedStart)} - ${this.core.formatTime(this.core.selectedEnd)} (${this.core.formatTime(duration)})`;
            } else {
                this.core.ui.selectionDisplay.textContent = 'No selection';
            }
        }
    }
    
    // Show message
    showMessage(message) {
        if (this.core.ui.messageDisplay) {
            this.core.ui.messageDisplay.textContent = message;
            this.core.ui.messageDisplay.style.color = '#4a9eff';
            
            // Clear message after 3 seconds
            setTimeout(() => {
                if (this.core.ui.messageDisplay) {
                    this.core.ui.messageDisplay.textContent = '';
                }
            }, 3000);
        }
    }
    
    // Update status
    updateStatus(status) {
        if (this.core.ui.statusDisplay) {
            this.core.ui.statusDisplay.textContent = status;
        }
    }
}