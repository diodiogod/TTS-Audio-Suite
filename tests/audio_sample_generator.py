"""
CosyVoice3 Audio Sample Generator

Generates audio samples with different parameter combinations
to help identify quality issues and optimal settings.

Usage:
    python tests/audio_sample_generator.py [--server-url URL]

Outputs:
    tests/output/audio_samples/
    ├── cosyvoice_zero_shot_speed1.0.wav
    ├── cosyvoice_instruct_gentle.wav
    ├── cosyvoice_cross_lingual.wav
    └── generation_log.csv
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("❌ requests not installed. Run: pip install requests")
    sys.exit(1)


class AudioSampleGenerator:
    """Generates audio samples via ComfyUI API for parameter testing."""
    
    # Voice sample to use for testing
    DEFAULT_VOICE = "voices_examples/Sophie_Anderson CC3.wav"
    
    # Test text samples (kept short for faster generation)
    TEST_TEXTS = {
        "english_short": "Hello, this is a test of the text to speech system.",
        "english_medium": "The quick brown fox jumps over the lazy dog. This is a pangram that contains every letter of the alphabet.",
        "chinese_short": "你好，这是一个测试。",
    }
    
    # Parameter combinations to test
    TEST_CASES = [
        # Mode tests
        {"name": "zero_shot_default", "mode": "zero_shot", "speed": 1.0, "text": "english_short"},
        {"name": "zero_shot_slow", "mode": "zero_shot", "speed": 0.7, "text": "english_short"},
        {"name": "zero_shot_fast", "mode": "zero_shot", "speed": 1.3, "text": "english_short"},
        {"name": "zero_shot_medium_text", "mode": "zero_shot", "speed": 1.0, "text": "english_medium"},
        
        # Cross-lingual mode (no reference text needed)
        {"name": "cross_lingual_english", "mode": "cross_lingual", "speed": 1.0, "text": "english_short"},
        
        # Instruct mode tests (Chinese instructions)
        {"name": "instruct_gentle", "mode": "instruct", "speed": 1.0, "text": "english_short", 
         "instruct_text": "请用温柔的语气说。"},
        {"name": "instruct_fast", "mode": "instruct", "speed": 1.0, "text": "english_short",
         "instruct_text": "请用尽可能快地语速说一句话。"},
    ]
    
    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.server_url = server_url
        self.output_dir = PROJECT_ROOT / "tests" / "output" / "audio_samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def check_server(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            r = requests.get(f"{self.server_url}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False
    
    def create_workflow(self, test_case: dict) -> dict:
        """Create workflow JSON for a test case."""
        text = self.TEST_TEXTS[test_case["text"]]
        mode = test_case["mode"]
        speed = test_case.get("speed", 1.0)
        instruct_text = test_case.get("instruct_text", "")
        
        # Base workflow
        workflow = {
            "1": {
                "class_type": "CosyVoiceEngineNode",
                "inputs": {
                    "model_path": "local:Fun-CosyVoice3-0.5B",
                    "device": "auto",
                    "mode": mode,
                    "speed": speed,
                    "use_fp16": True,
                }
            },
            "2": {
                "class_type": "UnifiedTTSTextNode",
                "inputs": {
                    "TTS_engine": ["1", 0],
                    "text": text,
                    "narrator_voice": self.DEFAULT_VOICE,
                    "seed": 42
                }
            },
            "3": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["2", 0],
                    "filename_prefix": f"cosyvoice_{test_case['name']}"
                }
            }
        }
        
        # Add optional parameters
        if mode == "zero_shot":
            # Reference text is loaded from .reference.txt by the system
            workflow["1"]["inputs"]["reference_text"] = ""  # Let system load from file
        
        if mode == "instruct" and instruct_text:
            workflow["1"]["inputs"]["instruct_text"] = instruct_text
            
        return workflow
    
    def execute_workflow(self, workflow: dict, timeout: int = 120) -> dict:
        """Execute workflow via ComfyUI API."""
        # Queue the prompt
        response = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow},
            timeout=10
        )
        
        if response.status_code != 200:
            error = response.json().get("error", {})
            return {"success": False, "error": error}
        
        prompt_id = response.json().get("prompt_id")
        if not prompt_id:
            return {"success": False, "error": "No prompt_id returned"}
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            history = requests.get(f"{self.server_url}/history/{prompt_id}", timeout=10)
            if history.status_code == 200:
                data = history.json()
                if prompt_id in data:
                    prompt_data = data[prompt_id]
                    if prompt_data.get("status", {}).get("completed"):
                        return {"success": True, "prompt_id": prompt_id, "data": prompt_data}
                    if prompt_data.get("status", {}).get("status_str") == "error":
                        return {"success": False, "error": prompt_data.get("status", {}).get("messages", [])}
            time.sleep(2)
        
        return {"success": False, "error": "Timeout waiting for completion"}
    
    def run_test_case(self, test_case: dict) -> dict:
        """Run a single test case and return results."""
        name = test_case["name"]
        print(f"\n🎵 Testing: {name}")
        print(f"   Mode: {test_case['mode']}, Speed: {test_case.get('speed', 1.0)}")
        
        start_time = time.time()
        
        try:
            workflow = self.create_workflow(test_case)
            result = self.execute_workflow(workflow)
            
            elapsed = time.time() - start_time
            
            if result["success"]:
                print(f"   ✅ Generated in {elapsed:.1f}s")
                return {
                    "name": name,
                    "success": True,
                    "duration_sec": elapsed,
                    "mode": test_case["mode"],
                    "speed": test_case.get("speed", 1.0),
                    "text": test_case["text"],
                    "error": ""
                }
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                return {
                    "name": name,
                    "success": False,
                    "duration_sec": elapsed,
                    "mode": test_case["mode"],
                    "speed": test_case.get("speed", 1.0),
                    "text": test_case["text"],
                    "error": str(result.get("error", "Unknown"))
                }
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Exception: {e}")
            return {
                "name": name,
                "success": False,
                "duration_sec": elapsed,
                "mode": test_case["mode"],
                "speed": test_case.get("speed", 1.0),
                "text": test_case["text"],
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all test cases and save results."""
        print("=" * 60)
        print("CosyVoice3 Audio Sample Generator")
        print("=" * 60)
        
        if not self.check_server():
            print(f"❌ ComfyUI server not running at {self.server_url}")
            print("   Start ComfyUI first, then run this script again.")
            return False
        
        print(f"✅ Connected to ComfyUI at {self.server_url}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎤 Voice sample: {self.DEFAULT_VOICE}")
        print(f"\n🧪 Running {len(self.TEST_CASES)} test cases...\n")
        
        for test_case in self.TEST_CASES:
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Save results to CSV
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return True
    
    def save_results(self):
        """Save results to CSV file."""
        csv_path = self.output_dir / "generation_log.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "success", "duration_sec", "mode", "speed", "text", "error", "quality_rating"
            ])
            writer.writeheader()
            for result in self.results:
                result["quality_rating"] = ""  # User fills this in
                writer.writerow(result)
        
        print(f"\n📝 Results saved to: {csv_path}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - successful
        
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"\n📂 Audio files saved to: {self.output_dir}")
        print(f"   (Look in ComfyUI output folder for SaveAudio outputs)")
        print(f"\n📝 Review log: {self.output_dir / 'generation_log.csv'}")
        print("\n👂 Please listen to each audio file and rate quality in the CSV!")


def main():
    parser = argparse.ArgumentParser(description="Generate CosyVoice3 audio samples")
    parser.add_argument("--server-url", default="http://127.0.0.1:8188", help="ComfyUI server URL")
    args = parser.parse_args()
    
    generator = AudioSampleGenerator(server_url=args.server_url)
    generator.run_all_tests()


if __name__ == "__main__":
    main()
