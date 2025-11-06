***
date: <yyyy-mm-dd>
agent: <agent/version/hardware>
environment:
  python: <ver>
  os: <os/version>
  deps: [list]
errors:
  - <error_id> (ref to troubleshooting)
notes:
  - <keep fatal/recurring/slow/flaky>
pitfalls:
  - <text>
replicable_setup:
  - <step/checklist>
***

# REPLICATION NOTES

## Environment Details
- **Date:** 2025-11-06
- **Agent:** Cline AI Assistant
- **Hardware:** macOS with M-series chip
- **Python:** 3.10+ recommended (3.14 has MediaPipe compatibility issues)

## Known Pitfalls
- MediaPipe requires Python 3.8-3.11; avoid Python 3.12+
- Virtual environment activation required for all operations
- Video files must be in MP4 format with H.264 codec
- Face detection works best with frontal lighting

## Replicable Setup Checklist
- [ ] Clone repository: `git clone https://github.com/MrSnowNB/redesigned-octo-barnacle.git`
- [ ] Create venv: `python3.10 -m venv venv`
- [ ] Activate: `source venv/bin/activate`
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Place video: `cp your_video.mp4 input/base_video_static.mp4`
- [ ] Configure timestamps in `config/viseme_config.json`
- [ ] Run extraction: `python run_pipeline.py extract`
- [ ] Test blending: `python run_pipeline.py blend`
