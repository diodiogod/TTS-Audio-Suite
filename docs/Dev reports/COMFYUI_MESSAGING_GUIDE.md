# ComfyUI Python-JavaScript Communication Guide

This document contains findings from attempting to implement ComfyUI's messaging system for the Audio Wave Analyzer node widget removal.

## Background

**Goal**: Remove the visible `node_id` widget from Audio Wave Analyzer while maintaining communication between Python backend and JavaScript frontend for visualization data.

**Previous Working Approach**: Used a visible widget that stored the node ID, Python read it to create temp files, JavaScript knew which file to fetch.

## ComfyUI Messaging System Documentation

### From ComfyUI Documentation Links:

- https://github.com/chrisgoringe/Comfy-Custom-Node-How-To/wiki/messages
- https://github.com/chrisgoringe/Comfy-Custom-Node-How-To/wiki/Passing-control-to-javascript

### Key Concepts Learned:

1. **Python to JavaScript Communication**:
   
   ```python
   from server import PromptServer
   dictionary_of_stuff = {"something": "A text message"}
   PromptServer.instance.send_sync("my-message-handle", dictionary_of_stuff)
   ```

2. **JavaScript Message Handling**:
   
   ```javascript
   import { app } from "../../../scripts/app.js";
   def myMessageHandler(event) {
       alert(event.detail.something);
   }
   // in setup()
   api.addEventListener("my-message-handle", myMessageHandler);
   ```

3. **Message Format**: 
   
   - Uses unique message handles
   - Sends JSON-serializable dictionaries
   - Event-based communication
   - Provides a way to pass data without using widgets

4. **WebSocket Interception**:
   
   - Messages flow through ComfyUI's WebSocket connection
   - Can be intercepted by hooking into `api.socket.onmessage`
   - Custom message types can be handled before ComfyUI's default handler

## Implementation Attempts and Failures

### Attempt 1: Standard Event Listener Approach

**Code**: Used `window.api.addEventListener("audio_analyzer_data", handler)`

**Result**: FAILED

- Messages were sent from Python: `🎵 Audio data sent via message to node X`
- Browser showed: `Unhandled message: {"type": "audio_analyzer_data", "data": {...}}`
- Error: `Unknown message type audio_analyzer_data`
- **Problem**: ComfyUI's default handler processed message first, threw error before our listener could catch it

### Attempt 2: WebSocket Message Interception (Late Hook)

**Code**: Hooked `window.api.socket.onmessage` in `beforeRegisterNodeDef()`

**Result**: FAILED

- Same error as Attempt 1
- **Problem**: Hook was established too late, after ComfyUI's handlers were already set up

### Attempt 3: Early WebSocket Interception

**Code**: Moved hook to `setup()` method to intercept earlier

**Result**: CATASTROPHIC FAILURE

- **ComfyUI interface completely failed to load**
- **Problem**: Interfered with ComfyUI's core WebSocket communication during initialization

### Attempt 4: WebSocket addEventListener (Non-Destructive)

**Code**: Used `addEventListener` instead of replacing `onmessage`

**Result**: FAILED

- Messages still went to ComfyUI's default handler first
- Still got "Unknown message type" errors
- **Problem**: `addEventListener` doesn't prevent default handler execution

## Technical Findings

### Message Flow Analysis:

1. **Python**: `PromptServer.instance.send_sync("audio_analyzer_data", data)` ✅ Works
2. **WebSocket**: Message arrives at browser via WebSocket ✅ Works  
3. **ComfyUI Handler**: Processes message, throws "Unknown message type" error ❌ Blocks us
4. **Our Handler**: Never gets called ❌ Fails

### Key Issues Discovered:

1. **Handler Priority**: ComfyUI's default WebSocket handler runs first and throws errors for unknown message types
2. **Message Type Registration**: ComfyUI may require custom message types to be registered somewhere
3. **Timing Issues**: Setting up interceptors too early breaks ComfyUI, too late doesn't work
4. **No Official Custom Message Support**: ComfyUI's messaging system might be designed only for internal use

### Browser Console Evidence:

```
api.ts:484 Unhandled message: {"type": "audio_analyzer_data", "data": {"node_id": "18", ...}}
Error: Unknown message type audio_analyzer_data at WebSocket.<anonymous> (api.ts:479:23)
```

## Why Temp File Approach Worked

The original widget-based temp file approach worked because:

1. **No Message Interception Needed**: Used standard HTTP fetch requests
2. **Predictable File Names**: Widget provided reliable node ID for filename
3. **Standard Web Access**: ComfyUI's `/temp/` endpoint is designed for file access
4. **No Timing Issues**: Files persist, can be fetched when ready
5. **No Handler Conflicts**: Doesn't interfere with ComfyUI's message system

## Conclusions and Recommendations

### What We Learned:

- ComfyUI's messaging system exists but may not support custom message types easily
- WebSocket interception is extremely risky and can break ComfyUI entirely
- The "Unknown message type" error suggests ComfyUI validates message types strictly
- Early interception breaks initialization, late interception doesn't work

### For Future Attempts:

1. **Research Message Type Registration**: Find if/how to register custom message types with ComfyUI
2. **Study ComfyUI Source**: Look at how internal messages are handled in ComfyUI's codebase
3. **Alternative Communication**: Consider using HTTP endpoints instead of WebSocket messages
4. **Plugin Architecture**: Look into ComfyUI's plugin/extension system for proper communication patterns

### Safe Alternatives:

1. **HTTP Endpoint**: Create custom HTTP endpoint that JavaScript can poll
2. **Hidden Widget**: Make widget truly invisible through CSS or DOM manipulation
3. **Global State**: Use global JavaScript variables set during node creation
4. **File System**: Stick with temp files but improve the ID synchronization

## Status: FAILED - Messaging Approach Not Viable

**Recommendation**: Revert to last working commit and explore alternative communication methods that don't interfere with ComfyUI's core message handling system.

**Last Working State**: Widget-based approach with temp files using node ID from widget value.