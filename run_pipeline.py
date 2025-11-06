"""
run_pipeline.py

Main Pipeline Orchestrator
Complete Alice Avatar RICo Layer processing pipeline

Usage:
    python run_pipeline.py extract    # Run viseme extraction
    python run_pipeline.py blend      # Run blending test
    python run_pipeline.py test       # Run integration test
    python run_pipeline.py all        # Run complete pipeline
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd: list, description: str):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if all required files exist"""
    required_files = [
        'requirements.txt',
        'config/viseme_config.json',
        'config/blending_config.json',
        'config/phoneme_map.json',
        'input/base_video_static.mp4',
        'src/01_extract_visemes.py',
        'src/02_rico_blender.py',
        'src/03_integration_test.py'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False

    print("✅ All required files present")
    return True

def run_extraction():
    """Run viseme extraction"""
    cmd = [sys.executable, 'src/01_extract_visemes.py']
    return run_command(cmd, "Viseme Extraction")

def run_blending_test():
    """Run blending test"""
    cmd = [sys.executable, 'src/02_rico_blender.py',
           '--text', 'Hello, world!',
           '--output', 'output/test_videos/test_output.mp4']
    return run_command(cmd, "RICo Blending Test")

def run_integration_test():
    """Run integration test"""
    cmd = [sys.executable, 'src/03_integration_test.py',
           '--text', 'Testing the complete pipeline']
    return run_command(cmd, "Integration Test")

def run_complete_pipeline():
    """Run the complete pipeline"""
    print("\n🚀 Starting Complete Alice Avatar RICo Pipeline")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        print("❌ Pipeline aborted due to missing dependencies")
        return False

    # Step 1: Extraction
    if not run_extraction():
        print("❌ Pipeline failed at extraction stage")
        return False

    # Step 2: Blending test
    if not run_blending_test():
        print("❌ Pipeline failed at blending stage")
        return False

    # Step 3: Integration test
    if not run_integration_test():
        print("❌ Pipeline failed at integration test")
        return False

    print("\n🎉 Pipeline completed successfully!")
    print("Check output/ directory for results")
    return True

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Alice Avatar RICo Pipeline')
    parser.add_argument('command', choices=['extract', 'blend', 'test', 'all'],
                       help='Pipeline command to run')

    args = parser.parse_args()

    if args.command == 'extract':
        success = run_extraction()
    elif args.command == 'blend':
        success = run_blending_test()
    elif args.command == 'test':
        success = run_integration_test()
    elif args.command == 'all':
        success = run_complete_pipeline()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
