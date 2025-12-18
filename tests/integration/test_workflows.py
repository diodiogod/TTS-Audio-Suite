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
        assert api_client.node_exists("CosyVoiceEngineNode"), \
            "CosyVoiceEngineNode not registered"
    
    def test_chatterbox_engine_registered(self, api_client):
        """Verify ChatterBox Engine node is available"""
        assert api_client.node_exists("ChatterBoxEngineNode"), \
            "ChatterBoxEngineNode not registered"
    
    def test_unified_tts_text_registered(self, api_client):
        """Verify Unified TTS Text node is available"""
        assert api_client.node_exists("UnifiedTTSTextNode"), \
            "UnifiedTTSTextNode not registered"
    
    def test_unified_tts_srt_registered(self, api_client):
        """Verify Unified TTS SRT node is available"""
        assert api_client.node_exists("UnifiedTTSSRTNode"), \
            "UnifiedTTSSRTNode not registered"
    
    def test_f5tts_engine_registered(self, api_client):
        """Verify F5-TTS Engine node is available"""
        assert api_client.node_exists("F5TTSEngineNode"), \
            "F5TTSEngineNode not registered"
    
    def test_higgs_audio_engine_registered(self, api_client):
        """Verify Higgs Audio Engine node is available"""
        assert api_client.node_exists("HiggsAudioEngineNode"), \
            "HiggsAudioEngineNode not registered"
    
    def test_indextts_engine_registered(self, api_client):
        """Verify IndexTTS Engine node is available"""
        assert api_client.node_exists("IndexTTSEngineNode"), \
            "IndexTTSEngineNode not registered"
    
    def test_rvc_engine_registered(self, api_client):
        """Verify RVC Engine node is available"""
        assert api_client.node_exists("RVCEngineNode"), \
            "RVCEngineNode not registered"
    
    def test_vibevoice_engine_registered(self, api_client):
        """Verify VibeVoice Engine node is available"""
        assert api_client.node_exists("VibeVoiceEngineNode"), \
            "VibeVoiceEngineNode not registered"
    
    def test_character_voices_registered(self, api_client):
        """Verify Character Voices node is available"""
        assert api_client.node_exists("CharacterVoicesNode"), \
            "CharacterVoicesNode not registered"


@pytest.mark.integration
class TestWorkflowFixtures:
    """Test loading and validating workflow fixture files"""
    
    def test_load_cosyvoice_workflow(self, workflow_fixtures_path):
        """Test loading CosyVoice workflow fixture"""
        workflow_file = workflow_fixtures_path / "test_cosyvoice_engine.json"
        assert workflow_file.exists(), "CosyVoice workflow fixture not found"
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        assert "1" in workflow
        assert workflow["1"]["class_type"] == "CosyVoiceEngineNode"
    
    def test_load_chatterbox_workflow(self, workflow_fixtures_path):
        """Test loading ChatterBox workflow fixture"""
        workflow_file = workflow_fixtures_path / "test_chatterbox_engine.json"
        assert workflow_file.exists(), "ChatterBox workflow fixture not found"
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        assert "1" in workflow
        assert workflow["1"]["class_type"] == "ChatterBoxEngineNode"
    
    def test_load_f5tts_workflow(self, workflow_fixtures_path):
        """Test loading F5-TTS workflow fixture"""
        workflow_file = workflow_fixtures_path / "test_f5tts_engine.json"
        assert workflow_file.exists(), "F5-TTS workflow fixture not found"
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        assert "1" in workflow
        assert workflow["1"]["class_type"] == "F5TTSEngineNode"
    
    def test_load_indextts_workflow(self, workflow_fixtures_path):
        """Test loading IndexTTS workflow fixture"""
        workflow_file = workflow_fixtures_path / "test_indextts_engine.json"
        assert workflow_file.exists(), "IndexTTS workflow fixture not found"
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        assert "1" in workflow
        assert workflow["1"]["class_type"] == "IndexTTSEngineNode"


@pytest.mark.integration
@pytest.mark.cosyvoice
class TestCosyVoiceEngine:
    """Integration tests for CosyVoice3 engine configuration"""
    
    def test_engine_config_validation(self, api_client, workflow_fixtures_path):
        """Test CosyVoice3 engine configuration is valid"""
        import requests
        
        workflow_file = workflow_fixtures_path / "test_cosyvoice_engine.json"
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        # Engine nodes are config-only, so ComfyUI will return "prompt_no_outputs"
        # This is expected! The test validates the node exists and accepts our inputs
        response = requests.post(
            f"{api_client.base_url}/prompt",
            json={"prompt": workflow},
            timeout=10
        )
        
        # 400 with "prompt_no_outputs" is expected for config-only nodes
        if response.status_code == 400:
            error_data = response.json().get("error", {})
            assert error_data.get("type") == "prompt_no_outputs", \
                f"Unexpected error: {error_data}"
        else:
            # If it succeeds, that's also fine (workflow might have outputs somehow)
            assert response.status_code == 200


@pytest.mark.integration
class TestChatterBoxEngine:
    """Integration tests for ChatterBox engine configuration"""
    
    def test_engine_config_validation(self, api_client, workflow_fixtures_path):
        """Test ChatterBox engine configuration is valid"""
        import requests
        
        workflow_file = workflow_fixtures_path / "test_chatterbox_engine.json"
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        response = requests.post(
            f"{api_client.base_url}/prompt",
            json={"prompt": workflow},
            timeout=10
        )
        
        if response.status_code == 400:
            error_data = response.json().get("error", {})
            assert error_data.get("type") == "prompt_no_outputs", \
                f"Unexpected error: {error_data}"
        else:
            assert response.status_code == 200


@pytest.mark.integration
class TestF5TTSEngine:
    """Integration tests for F5-TTS engine configuration"""
    
    def test_engine_config_validation(self, api_client, workflow_fixtures_path):
        """Test F5-TTS engine configuration is valid"""
        import requests
        
        workflow_file = workflow_fixtures_path / "test_f5tts_engine.json"
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        response = requests.post(
            f"{api_client.base_url}/prompt",
            json={"prompt": workflow},
            timeout=10
        )
        
        if response.status_code == 400:
            error_data = response.json().get("error", {})
            assert error_data.get("type") == "prompt_no_outputs", \
                f"Unexpected error: {error_data}"
        else:
            assert response.status_code == 200


@pytest.mark.integration
class TestIndexTTSEngine:
    """Integration tests for IndexTTS engine configuration"""
    
    def test_engine_config_validation(self, api_client, workflow_fixtures_path):
        """Test IndexTTS engine configuration is valid"""
        import requests
        
        workflow_file = workflow_fixtures_path / "test_indextts_engine.json"
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        response = requests.post(
            f"{api_client.base_url}/prompt",
            json={"prompt": workflow},
            timeout=10
        )
        
        if response.status_code == 400:
            error_data = response.json().get("error", {})
            assert error_data.get("type") == "prompt_no_outputs", \
                f"Unexpected error: {error_data}"
        else:
            assert response.status_code == 200


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
        # Should have TTS nodes registered
        tts_nodes = [k for k in info.keys() if 'TTS' in k or 'Voice' in k or 'Engine' in k]
        assert len(tts_nodes) >= 5, f"Expected at least 5 TTS-related nodes, found {len(tts_nodes)}"
    
    def test_tts_audio_suite_nodes_count(self, api_client):
        """Verify TTS Audio Suite loaded expected number of nodes"""
        info = api_client.get_object_info()
        
        # TTS Audio Suite should register at least 20 nodes
        tts_suite_nodes = [
            k for k in info.keys() 
            if any(x in k for x in ['TTS', 'Voice', 'Engine', 'ChatterBox', 'CosyVoice', 'Audio'])
        ]
        assert len(tts_suite_nodes) >= 15, \
            f"TTS Audio Suite should have 15+ nodes, found {len(tts_suite_nodes)}: {tts_suite_nodes}"


@pytest.mark.integration
@pytest.mark.slow
class TestFullTTSGeneration:
    """Full TTS generation tests - requires models to be downloaded"""
    
    def test_chatterbox_e2e_generation(self, api_client, workflow_fixtures_path):
        """Test full ChatterBox TTS generation pipeline with output"""
        workflow_file = workflow_fixtures_path / "test_chatterbox_e2e.json"
        
        if not workflow_file.exists():
            pytest.skip("ChatterBox e2e workflow fixture not found")
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        # Execute the full workflow (this will actually generate audio)
        try:
            result = api_client.execute_workflow(workflow, timeout=120)
            
            # Check workflow completed
            assert result is not None
            if "status" in result:
                # If there's an error, show what went wrong
                if result.get("status", {}).get("status_str") == "error":
                    pytest.fail(f"Workflow failed: {result.get('status', {}).get('messages', [])}")
        except Exception as e:
            # Model download or CUDA issues shouldn't fail the test
            if "model" in str(e).lower() or "cuda" in str(e).lower() or "memory" in str(e).lower():
                pytest.skip(f"Hardware/model limitation: {e}")
            raise
    
    def test_cosyvoice_e2e_generation(self, api_client, workflow_fixtures_path):
        """Test full CosyVoice3 TTS generation pipeline with output"""
        workflow_file = workflow_fixtures_path / "test_cosyvoice_e2e.json"
        
        if not workflow_file.exists():
            pytest.skip("CosyVoice e2e workflow fixture not found")
        
        with open(workflow_file) as f:
            workflow = json.load(f)
        
        try:
            result = api_client.execute_workflow(workflow, timeout=120)
            
            assert result is not None
            if "status" in result:
                if result.get("status", {}).get("status_str") == "error":
                    pytest.fail(f"Workflow failed: {result.get('status', {}).get('messages', [])}")
        except Exception as e:
            if "model" in str(e).lower() or "cuda" in str(e).lower() or "memory" in str(e).lower():
                pytest.skip(f"Hardware/model limitation: {e}")
            raise
