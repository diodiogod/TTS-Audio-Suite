# F5-TTS Testing Guide for ComfyUI

This guide will help you test the F5-TTS integration in ComfyUI manually.

## Prerequisites

Before testing, ensure you have:

1. **F5-TTS dependencies installed** (already included in main requirements):
   ```bash
   pip install -r requirements.txt
   ```

2. **F5-TTS models downloaded** (optional - will auto-download):
   - Download models from [HuggingFace F5-TTS models](https://huggingface.co/SWivid/F5-TTS)
   - Place in `ComfyUI/models/F5-TTS/` folder structure as shown in README.md
   - Models will auto-download from HuggingFace if not found locally

3. **ComfyUI running** with this extension installed

## Step 1: Start ComfyUI and Check Node Loading

1. **Start ComfyUI** from your ComfyUI directory
2. **Look for the startup messages** in the console:
   - You should see: `🚀 ChatterBox Voice Extension v2.0.2 loaded with X nodes`
   - Check if "🎤 F5-TTS Voice Generation" appears in the node list

## Step 2: Verify F5-TTS Node is Available

1. **In ComfyUI interface**, right-click to add a node
2. **Navigate to**: `Add Node` > `F5-TTS Voice` 
3. **Look for**: "🎤 F5-TTS Voice Generation"
4. **If the node appears**: F5-TTS integration is successful!
5. **If the node is missing**: Check console for error messages

## Step 3: Test Basic F5-TTS Generation

### Required Inputs:
1. **Reference Audio**: A short (5-30 second) clean audio file in WAV format
2. **Reference Text**: The exact text spoken in the reference audio
3. **Target Text**: The text you want to synthesize

### Testing Steps:

1. **Add the F5-TTS node** to your workflow
2. **Configure the node**:
   - **text**: "Hello! This is a test of F5-TTS integration with ChatterBox Voice."
   - **ref_text**: "This should match your reference audio exactly"
   - **model**: "F5TTS_Base" (default)
   - **device**: "auto"
   - **enable_chunking**: true (for longer texts)

3. **Connect reference audio**:
   - Use a `Load Audio` node to load your reference file
   - Connect it to the `reference_audio` input

4. **Add audio output**:
   - Connect an `Audio Preview` or `Save Audio` node to the output

5. **Queue the workflow** and check results

## Step 4: Test Different Models

Try different F5-TTS models to verify they load correctly:

- **F5TTS_Base**: English base model
- **F5TTS_v1_Base**: English v1 model  
- **E2TTS_Base**: E2-TTS model
- **F5-DE**: German model (if available)
- **F5-ES**: Spanish model (if available)

## Step 5: Test Text Chunking

For long text processing:

1. **Enable chunking**: Set `enable_chunking` to true
2. **Set chunk size**: Try `max_chars_per_chunk` = 300
3. **Test combination methods**:
   - "auto": Automatic selection
   - "concatenate": Simple joining
   - "silence_padding": Silence between chunks
   - "crossfade": Smooth transitions

## Expected Results

### Success Indicators:
- ✅ Node loads without errors
- ✅ Audio generates successfully
- ✅ Voice matches reference audio characteristics
- ✅ Text chunking works for long texts
- ✅ Different models can be selected

### Common Issues and Solutions:

#### "F5-TTS support not available"
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Install F5-TTS core: `pip install f5-tts` (or from source if needed)
- Restart ComfyUI

#### "Model loading failed"
- Check internet connection (for HuggingFace download)
- Verify model name is correct
- Try a different model

#### "Reference text required"
- Ensure `ref_text` field is not empty
- Reference text should match reference audio content

#### Audio quality issues
- Use high-quality reference audio (24kHz recommended)
- Ensure reference text accurately matches audio
- Adjust temperature and cfg_strength parameters

## Advanced Testing

### Custom Model Testing:
1. Place F5-TTS models in `ComfyUI/models/F5-TTS/`
2. Restart ComfyUI
3. Models should appear in the dropdown

### Performance Testing:
1. Test with different text lengths
2. Monitor memory usage
3. Test CPU vs GPU performance

## Troubleshooting Console Output

Look for these messages in the ComfyUI console:

### Success Messages:
```
✅ F5-TTS model 'F5TTS_Base' loaded successfully
📦 Loading F5-TTS model 'F5TTS_Base' from HuggingFace
🎤 Generating F5-TTS chunk 1/3...
```

### Error Messages:
```
❌ F5-TTS not available - missing dependencies
⚠️ Failed to load F5-TTS model from huggingface: [error details]
F5-TTS generation failed: [error details]
```

## Getting Help

If you encounter issues:

1. **Check the console output** for specific error messages
2. **Verify F5-TTS installation** by running: `python -c "import f5_tts"`
3. **Check dependencies** are installed correctly
4. **Review the integration guide** in `f5tts_integration_guide.py`

## Example Workflow Structure

```
[Load Audio] -> [🎤 F5-TTS Voice Generation] -> [Preview Audio]
             -> (reference_audio)              -> (audio)
             
[Text Input] -> (text)
[Text Input] -> (ref_text)
```

This basic workflow will help you verify F5-TTS is working correctly.