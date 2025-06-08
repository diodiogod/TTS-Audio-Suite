# ComfyUI Testing Guide for ChatterBox SRT Fix

## 🎯 Purpose
This guide helps you test the ComfyUI loading fixes for the ChatterBox SRT implementation in a real ComfyUI environment.

## 🔧 Pre-Test Setup

### 1. Backup Current Installation
```bash
# Backup your current nodes.py if you have modifications
cp nodes.py nodes.py.backup
```

### 2. Verify File Structure
Ensure your ComfyUI custom nodes directory has:
```
ComfyUI/custom_nodes/ComfyUI_ChatterBox_Voice/
├── nodes.py                    # ✅ Fixed version
├── chatterbox/
│   ├── __init__.py
│   ├── srt_parser.py          # ✅ SRT parser module
│   ├── audio_timing.py        # ✅ Audio timing module
│   ├── tts.py
│   └── vc.py
├── requirements.txt
└── README.md
```

## 🧪 Test Scenarios

### Test 1: ComfyUI Startup (Critical)
**Expected Result**: ComfyUI starts without import errors

1. Start ComfyUI normally
2. Check console output for:
   - ✅ `🎉 CHATTERBOX VOICE NODES LOADED SUCCESSFULLY!`
   - ✅ No `NameError: name 'SRTSubtitle' is not defined`
   - ✅ No `No module named 'chatterbox.srt_parser'`

**Console Output Should Show**:
```
🔍 Attempting to import ChatterBox modules...
📦 Attempting SRT imports from bundled chatterbox...
✅ ChatterboxTTSNode class defined
✅ ChatterboxVCNode class defined
✅ ChatterboxSRTTTSNode class defined
```

### Test 2: Node Availability
**Expected Result**: Nodes appear in ComfyUI interface

1. Right-click in ComfyUI workflow area
2. Navigate to: `Add Node` → `ChatterBox Voice`
3. Verify available nodes:
   - ✅ `🎤 ChatterBox Voice TTS`
   - ✅ `🔄 ChatterBox Voice Conversion`
   - ⚠️ `🎤 ChatterBox SRT Voice TTS` (only if dependencies installed)

### Test 3: Basic Node Functionality
**Expected Result**: Nodes can be added and configured

1. Add `🎤 ChatterBox Voice TTS` node
2. Verify input fields appear:
   - ✅ `text` (multiline)
   - ✅ `device` (auto/cuda/cpu)
   - ✅ `exaggeration`, `temperature`, `cfg_weight`
   - ✅ Optional chunking controls

3. Add `🔄 ChatterBox Voice Conversion` node
4. Verify input fields appear:
   - ✅ `source_audio`, `target_audio`
   - ✅ `device` selection

### Test 4: Error Handling (When Dependencies Missing)
**Expected Result**: Clear error messages, no crashes

1. Try to execute TTS node without ChatterBox installed
2. Expected error message:
   ```
   ChatterboxTTS not available - check installation or add bundled version
   ```

3. If SRT node appears, try to execute it
4. Expected error message:
   ```
   SRT support not available - missing required modules
   ```

## 🐛 Troubleshooting

### Issue: ComfyUI Won't Start
**Symptoms**: Import errors, crashes on startup
**Solution**:
1. Check ComfyUI console for specific error
2. Verify `nodes.py` has the latest fixes
3. Ensure `chatterbox/` directory exists with required files

### Issue: Nodes Don't Appear
**Symptoms**: ChatterBox nodes missing from menu
**Solution**:
1. Check console for node registration messages
2. Restart ComfyUI completely
3. Verify `NODE_CLASS_MAPPINGS` in console output

### Issue: SRT Node Missing
**Symptoms**: Only TTS and VC nodes appear
**Expected**: This is normal when SRT dependencies aren't installed
**To Enable SRT**:
```bash
pip install -r requirements.txt
# Or install specific dependencies as needed
```

## ✅ Success Criteria

### Minimum Success (Core Fix)
- ✅ ComfyUI starts without import errors
- ✅ ChatterBox TTS and VC nodes are available
- ✅ Nodes can be added to workflow without crashes
- ✅ Clear error messages when dependencies missing

### Full Success (With Dependencies)
- ✅ All above criteria met
- ✅ SRT TTS node also available
- ✅ SRT functionality works with test content
- ✅ All three node types function correctly

## 📊 Test Results Template

```
ComfyUI ChatterBox SRT Fix Test Results
=====================================
Date: ___________
ComfyUI Version: ___________
Python Version: ___________

Test 1 - ComfyUI Startup:           [ ] PASS [ ] FAIL
Test 2 - Node Availability:         [ ] PASS [ ] FAIL  
Test 3 - Basic Node Functionality:  [ ] PASS [ ] FAIL
Test 4 - Error Handling:            [ ] PASS [ ] FAIL

Nodes Available:
- ChatterBox Voice TTS:              [ ] YES [ ] NO
- ChatterBox Voice Conversion:       [ ] YES [ ] NO  
- ChatterBox SRT Voice TTS:          [ ] YES [ ] NO

Issues Found:
_________________________________________________
_________________________________________________

Overall Result:                      [ ] SUCCESS [ ] NEEDS WORK
```

## 🚀 Next Steps After Testing

### If Tests Pass
1. The fix is working correctly
2. SRT functionality will activate when dependencies are installed
3. You can use the existing TTS and VC nodes immediately

### If Tests Fail
1. Note specific error messages
2. Check which test scenario failed
3. Provide console output for further debugging

## 📞 Support Information

If you encounter issues during testing:
1. Capture the full ComfyUI console output
2. Note which specific test step failed
3. Include any error messages or stack traces
4. Specify your ComfyUI and Python versions

The fixes should resolve the original import errors and provide graceful degradation when dependencies are missing.