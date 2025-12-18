"""
Integration tests for TTS-Audio-Suite workflows
Tests execute actual workflows through ComfyUI API
"""

import pytest
import json
import sys
from pathlib import Path

# Add custom node root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
class TestNodeRegistration:
    """Test that all TTS nodes are properly registered"""
    
    def test_cosyvoice_engine_registered(self, api_client):
        """Verify CosyVoice Engine node is available"""
        assert api_client.node_exists("CosyVoice Engine"), \
            "CosyVoice Engine node not registered"
    
    def test_chatterbox_engine_registered(self, api_client):
        """Verify ChatterBox Engine node is available"""
        assert api_client.node_exists("ChatterBox Engine"), \
            "ChatterBox Engine node not registered"
    
    def test_unified_tts_text_registered(self, api_client):
        """Verify Unified TTS Text node is available"""
        assert api_client.node_exists("TTS - Text"), \
            "TTS - Text (Unified) node not registered"
    
    def test_unified_tts_srt_registered(self, api_client):
        """Verify Unified TTS SRT node is available"""
        assert api_client.node_exists("TTS - SRT"), \
            "TTS - SRT (Unified) node not registered"
    
    def test_f5tts_engine_registered(self, api_client):
        """Verify F5-TTS Engine node is available"""
        # F5-TTS may not be available on all systems
        if not api_client.node_exists("F5-TTS Engine"):
            pytest.skip("F5-TTS Engine not available")
    
    def test_higgs_audio_engine_registered(self, api_client):
        """Verify Higgs Audio Engine node is available"""
        if not api_client.node_exists("Higgs Audio Engine"):
            pytest.skip("Higgs Audio Engine not available")


@pytest.mark.integration
@pytest.mark.cosyvoice
class TestCosyVoiceEngine:
    """Integration tests for CosyVoice3 engine"""
    
    def test_engine_config_zero_shot(self, api_client, workflow_fixtures_path):
        """Test CosyVoice3 engine configuration in zero_shot mode"""
        workflow = {
            "1": {
                "class_type": "CosyVoice Engine",
                "inputs": {
                    "model_path": "Fun-CosyVoice3-0.5B",
                    "device": "auto",
                    "mode": "zero_shot",
                    "speed": 1.0,
                    "use_fp16": True,
                    "reference_text": "Hello, this is a test."
                }
            }
        }
        
        result = api_client.execute_workflow(workflow, timeout=60)
        assert result["status"]["completed"] == True
    
    def test_engine_config_instruct_mode(self, api_client):
        """Test CosyVoice3 engine in instruct mode"""
        workflow = {
            "1": {
                "class_type": "CosyVoice Engine",
                "inputs": {
                    "model_path": "Fun-CosyVoice3-0.5B",
                    "device": "auto",
                    "mode": "instruct",
                    "speed": 1.0,
                    "use_fp16": True,
                    "instruct_text": "请用温柔的语气说。"
                }
            }
        }
        
        result = api_client.execute_workflow(workflow, timeout=60)
        assert result["status"]["completed"] == True


@pytest.mark.integration
@pytest.mark.slow
class TestTTSGeneration:
    """Full TTS generation tests (slow, requires model loading)"""
    
    def test_cosyvoice_text_generation(self, api_client, sample_voice_path):
        """Test full CosyVoice3 TTS text generation pipeline"""
        if not sample_voice_path:
            pytest.skip("No sample voice file available")
        
        # This workflow would need a proper voice reference
        # For now, skip if no voice samples available
        pytest.skip("Full generation test requires voice samples - manual test recommended")
    
    def test_chatterbox_text_generation(self, api_client, sample_voice_path):
        """Test ChatterBox TTS text generation"""
        if not sample_voice_path:
            pytest.skip("No sample voice file available")
        
        pytest.skip("Full generation test requires voice samples - manual test recommended")


@pytest.mark.integration
class TestWorkflowFixtures:
    """Test loading and validating workflow fixture files"""
    
    def test_load_cosyvoice_test_workflow(self, workflow_fixtures_path):
        """Test loading CosyVoice test workflow fixture"""
        workflow_file = workflow_fixtures_path / "test_cosyvoice_engine.json"
        
        if not workflow_file.exists():
            pytest.skip("Workflow fixture not found")
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        assert isinstance(workflow, dict)
        assert len(workflow) > 0


@pytest.mark.integration
class TestServerHealth:
    """Basic server health checks"""
    
    def test_server_system_stats(self, api_client):
        """Test that we can get system stats"""
        stats = api_client.get_system_stats()
        
        assert "system" in stats
        assert "devices" in stats
    
    def test_server_object_info(self, api_client):
        """Test that we can get object info"""
        info = api_client.get_object_info()
        
        assert isinstance(info, dict)
        # Should have at least some nodes registered
        assert len(info) > 10
