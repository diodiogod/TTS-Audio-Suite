/**
 * Audio Wave Analyzer UI Module (Refactored)
 * Main coordinator that orchestrates all UI functionality through modular components
 */

import { AudioAnalyzerControls } from './audio_analyzer_controls.js';
import { AudioAnalyzerWidgets } from './audio_analyzer_widgets.js';
import { AudioAnalyzerLayout } from './audio_analyzer_layout.js';

export class AudioAnalyzerUI {
    constructor(core) {
        this.core = core;
        this.container = null;
        this.canvas = null;
        this.playButton = null;
        this.stopButton = null;
        this.timeDisplay = null;
        this.selectionDisplay = null;
        this.statusDisplay = null;
        this.messageDisplay = null;
        this.controlsContainer = null;
        this.widget = null;
        
        // Initialize modular components
        this.controlsModule = new AudioAnalyzerControls(core);
        this.widgets = new AudioAnalyzerWidgets(core);
        this.layout = new AudioAnalyzerLayout(core);
        
        // Make components available to core for cross-component communication
        this.core.controls = this.controlsModule;
        this.core.widgets = this.widgets;
        this.core.layout = this.layout;
    }
    
    createInterface() {
        // Create Audio Wave Analyzer UI interface using modular components
        
        // Remove existing interface
        const existingInterface = this.core.node.widgets?.find(w => w.name === 'audio_analyzer_interface');
        if (existingInterface) {
            const existingContainer = existingInterface.element;
            if (existingContainer && existingContainer.parentNode) {
                existingContainer.parentNode.removeChild(existingContainer);
            }
        }
        
        // Create main container using layout module
        this.container = this.layout.createMainContainer();
        
        // Create canvas using layout module
        this.canvas = this.layout.createCanvas();
        
        // Get canvas context
        this.core.canvas = this.canvas;
        this.core.ctx = this.canvas.getContext('2d');
        
        // Create controls container using layout module
        this.controlsContainer = this.layout.createControlsContainer();
        
        // Create UI components using controls module
        const playbackControls = this.controlsModule.createPlaybackControls();
        const speedSlider = this.controlsModule.createSpeedSlider();
        const mainControls = this.controlsModule.createMainControls();
        const zoomControls = this.controlsModule.createZoomControls();
        const statusDisplays = this.controlsModule.createStatusDisplays();
        
        // Add zoom controls to the first row
        playbackControls.appendChild(zoomControls);
        
        // Assemble controls
        this.controlsContainer.appendChild(playbackControls);
        this.controlsContainer.appendChild(mainControls);
        this.controlsContainer.appendChild(statusDisplays);
        
        // Assemble interface
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controlsContainer);
        
        // Add floating analyze button to canvas
        this.addFloatingAnalyzeButton();
        
        // Add floating speed slider to canvas
        this.addFloatingSpeedSlider();
        
        // Add container to node using layout module
        const success = this.layout.addContainerToNode(this.container);
        
        if (success) {
            // Insert the spacer widget to reserve space
            const spacerWidget = this.widgets.insertSpacerWidget();
            
            // Setup initial canvas size using controls module
            this.controlsModule.setupCanvasSize();
            
            // Setup canvas resize observer using controls module
            this.controlsModule.setupCanvasResize();
            
            // Setup drag and drop using controls module
            this.controlsModule.setupDragAndDrop();
            
            // console.log('🌊 Audio Wave Analyzer: Interface setup complete - spacer reserves space, interface positioned over it');  // Debug: setup complete
        }
    }
    
    addFloatingAnalyzeButton() {
        // Create floating analyze button
        const floatingAnalyzeButton = document.createElement('button');
        floatingAnalyzeButton.textContent = '🔍 Analyze';
        floatingAnalyzeButton.onclick = () => this.core.onParametersChanged();
        
        // Position it dead center of canvas for testing
        floatingAnalyzeButton.style.cssText = `
            position: absolute;
            top: -6.5%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 100;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            color: white;
            background: #28a745;
            box-shadow: 0 2px 8px rgba(40, 167, 69, 0.4);
        `;
        
        // Allow the container to show content outside its bounds
        this.container.style.overflow = 'visible';
        
        // Add to canvas container
        this.container.appendChild(floatingAnalyzeButton);
        
        // console.log('🌊 Audio Wave Analyzer: Added floating analyze button at canvas center');  // Debug: button placement
    }
    
    addFloatingSpeedSlider() {

        // Create floating speed slider container             backdrop-filter: blur(4px);
        const floatingSliderContainer = document.createElement('div');
        floatingSliderContainer.style.cssText = `
            position: absolute;
            bottom: 107px;
            left: 0;
            right: 0;
            z-index: 99;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;

        `;
        
        const speedLabel = document.createElement('span');
        speedLabel.textContent = 'Speed:';
        speedLabel.style.cssText = `
            color: #fff;
            font-size: 11px;
            font-weight: bold;
        `;
        
        this.core.ui.speedSlider = document.createElement('input');
        this.core.ui.speedSlider.type = 'range';
        this.core.ui.speedSlider.min = '0.00';
        this.core.ui.speedSlider.max = '2.00';
        this.core.ui.speedSlider.step = '0.025';
        this.core.ui.speedSlider.value = '1';
        this.core.ui.speedSlider.style.cssText = `
            flex: 1;
            height: 20px;
            background: transparent;
            outline: none;
            cursor: pointer;
            -webkit-appearance: none;
            appearance: none;
            border: none;
            position: relative;
            z-index: 2;
        `;
        
        // Simple CSS for just the thumb
        const thumbStyle = document.createElement('style');
        thumbStyle.textContent = `
            input[type="range"].speed-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 2px;
                height: 20px;
                background: white;
                border: none;
                border-radius: 0;
                cursor: pointer;
            }
            input[type="range"].speed-slider::-moz-range-thumb {
                -moz-appearance: none;
                width: 2px;
                height: 20px;
                background: white;
                border: none;
                border-radius: 0;
                cursor: pointer;
            }
        `;
        document.head.appendChild(thumbStyle);
        
        this.core.ui.speedSlider.className = 'speed-slider';
        
        // Create custom track line behind the slider
        const customTrack = document.createElement('div');
        customTrack.style.cssText = `
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: #666;
            transform: translateY(-50%);
            pointer-events: none;
            z-index: 1;
        `;
        
        this.core.ui.speedValue = document.createElement('span');
        this.core.ui.speedValue.textContent = '1.00x';
        this.core.ui.speedValue.style.cssText = `
            color: #fff;
            font-size: 11px;
            font-weight: bold;
            min-width: 40px;
            text-align: center;
        `;
        
        // Track extended dragging beyond slider limits
        let extendedSpeed = 1.0;
        let isDragging = false;
        let lastMouseX = 0;
        
        this.core.ui.speedSlider.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastMouseX = e.clientX;
            extendedSpeed = parseFloat(this.core.ui.speedSlider.value);
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const deltaX = e.clientX - lastMouseX;
            lastMouseX = e.clientX;
            
            // Calculate base sensitivity based on slider width
            const sliderRect = this.core.ui.speedSlider.getBoundingClientRect();
            const baseSensitivity = 2.0 / sliderRect.width; // 2.0 is the normal range (0.0 to 2.0)
            
            // Calculate how far we are from normal bounds for acceleration
            let acceleration = 1.0;
            if (extendedSpeed < 0.0) {
                // Left side acceleration: the more negative, the faster it goes
                const outOfBounds = Math.abs(extendedSpeed);
                acceleration = 1.0 + (outOfBounds * 3.0); // Much stronger linear acceleration
            } else if (extendedSpeed > 2.0) {
                // Right side acceleration: the higher above 2.0, the faster it goes
                const outOfBounds = extendedSpeed - 2.0;
                acceleration = 1.0 + (outOfBounds * 2.5); // Much stronger linear acceleration
            }
            
            // Apply accelerated sensitivity
            const sensitivity = baseSensitivity * acceleration;
            
            // Update extended speed based on mouse movement
            extendedSpeed += deltaX * sensitivity;
            
            // Clamp to absolute limits
            extendedSpeed = Math.max(-8.0, Math.min(8.0, extendedSpeed));
            
            // Update slider position (clamped to visual range)
            const clampedValue = Math.max(0.0, Math.min(2.0, extendedSpeed));
            this.core.ui.speedSlider.value = clampedValue;
            
            // Update display and speed with extended value
            this.core.ui.speedValue.textContent = `${extendedSpeed.toFixed(2)}x`;
            this.core.setPlaybackSpeed(extendedSpeed);
        });
        
        document.addEventListener('mouseup', () => {
            isDragging = false;
            // Force sync when drag ends to ensure consistency
            extendedSpeed = parseFloat(this.core.ui.speedSlider.value);
            this.core.ui.speedValue.textContent = `${extendedSpeed.toFixed(2)}x`;
            this.core.setPlaybackSpeed(extendedSpeed);
        });
        
        this.core.ui.speedSlider.oninput = () => {
            const currentSliderValue = parseFloat(this.core.ui.speedSlider.value);
            
            if (!isDragging) {
                // Normal slider input (not during extended drag)
                extendedSpeed = currentSliderValue;
            } else {
                // During extended drag, ensure we don't lose sync
                // If slider is at edge but we're dragging, keep extended value
                if ((currentSliderValue === 2.0 && extendedSpeed > 2.0) || 
                    (currentSliderValue === 0.0 && extendedSpeed < 0.0)) {
                    // Keep the extended speed, don't reset to slider value
                } else {
                    // We're within normal range, sync with slider
                    extendedSpeed = currentSliderValue;
                }
            }
            
            this.core.ui.speedValue.textContent = `${extendedSpeed.toFixed(2)}x`;
            this.core.setPlaybackSpeed(extendedSpeed);
        };
        
        // Create slider wrapper for proper positioning of track and slider
        const sliderWrapper = document.createElement('div');
        sliderWrapper.style.cssText = `
            position: relative;
            flex: 1;
            height: 20px;
            display: flex;
            align-items: center;
        `;
        
        sliderWrapper.appendChild(customTrack);
        sliderWrapper.appendChild(this.core.ui.speedSlider);
        
        floatingSliderContainer.appendChild(speedLabel);
        floatingSliderContainer.appendChild(sliderWrapper);
        floatingSliderContainer.appendChild(this.core.ui.speedValue);
        
        // Allow the container to show content outside its bounds
        this.container.style.overflow = 'visible';
        
        // Add to canvas container
        this.container.appendChild(floatingSliderContainer);
        
        // console.log('🌊 Audio Wave Analyzer: Added floating speed slider spanning canvas width');  // Debug: slider placement
    }
    
    // Delegate methods to appropriate modules
    
    // Time and selection display methods (delegated to controls)
    updateTimeDisplay() {
        this.controlsModule.updateTimeDisplay();
    }
    
    updateSelectionDisplay() {
        this.controlsModule.updateSelectionDisplay();
    }
    
    showMessage(message) {
        this.controlsModule.showMessage(message);
    }
    
    updateStatus(status) {
        this.controlsModule.updateStatus(status);
    }
    
    // Canvas methods (delegated to controls)
    setupCanvasSize() {
        this.controlsModule.setupCanvasSize();
    }
    
    setupCanvasResize() {
        this.controlsModule.setupCanvasResize();
    }
    
    // Layout methods (delegated to layout)
    resizeNodeForInterface() {
        this.layout.resizeNodeForInterface();
    }
    
    updateNodeLayout() {
        this.layout.updateNodeLayout();
    }
    
    setupNodeResizeHandling() {
        this.layout.setupNodeResizeHandling();
    }
    
    // Widget methods (delegated to widgets)
    setupWidgetHeight(widget) {
        this.widgets.setupWidgetHeight(widget);
    }
    
    insertSpacerWidget() {
        return this.widgets.insertSpacerWidget();
    }
    
    positionInterfaceOverSpacer() {
        this.widgets.positionInterfaceOverSpacer();
    }
    
    findInsertPosition() {
        return this.widgets.findInsertPosition();
    }
    
    setupMultilineWidgetWatchers() {
        this.widgets.setupMultilineWidgetWatchers();
    }
    
    recalculateNodeHeight() {
        this.widgets.recalculateNodeHeight();
    }
    
    ensureUIVisible() {
        this.widgets.ensureUIVisible();
    }
    
    // Cleanup
    destroy() {
        this.layout.destroy();
        console.log('🌊 Audio Wave Analyzer UI destroyed and cleaned up');
    }
}