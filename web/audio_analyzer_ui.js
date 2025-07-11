/**
 * Audio Analyzer UI Module (Refactored)
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
        // Create Audio Analyzer UI interface using modular components
        
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
            
            console.log('🎵 Audio Analyzer: Interface setup complete - spacer reserves space, interface positioned over it');
        }
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
        console.log('🎵 Audio Analyzer UI destroyed and cleaned up');
    }
}