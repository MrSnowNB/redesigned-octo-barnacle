# SETUP GUIDE
## Alice Avatar RICo Layer - Environment Configuration

---

## 🔧 **SYSTEM REQUIREMENTS**

### **Minimum Requirements:**
- **OS:** Windows 10/11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python:** 3.10 (recommended for MediaPipe compatibility)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 2GB free space for project + video files
- **CPU:** Multi-core processor (4+ cores recommended)

### **Optional (for performance):**
- **GPU:** NVIDIA GPU with CUDA support
- **GPU RAM:** 4GB+ VRAM
- **CUDA Toolkit:** 11.8+

---

## 📦 **INSTALLATION STEPS**

### **Step 1: Install Python**

```bash
# Check Python version
python --version
# Should be 3.9+

# If not installed, download from:
# https://www.python.org/downloads/
```

### **Step 2: Create Virtual Environment**

```bash
# Navigate to project directory
cd alice-rico-mvp

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import cv2, mediapipe, numpy; print('✓ All packages installed')"
```

---

## 📋 **REQUIREMENTS.TXT**

```txt
# Core dependencies
opencv-python==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
Pillow==10.1.0

# TTS and audio processing
pydub==0.25.1
azure-cognitiveservices-speech==1.32.1  # Optional: Azure TTS
phonemizer==3.2.1  # For phoneme extraction

# Utilities
tqdm==4.66.1
colorama==0.4.6

# Data handling
pandas==2.1.3
scipy==1.11.4

# Optional: GPU acceleration
# tensorflow-gpu==2.13.0  # Uncomment if using GPU
# torch==2.1.0+cu118  # Uncomment if using PyTorch with CUDA
```

---

## 🎬 **INITIAL SETUP**

### **Step 1: Create Project Structure**

```bash
# Create all necessary directories
mkdir -p input output/{visemes,metadata,test_videos,logs} src/utils config

# Verify structure
tree -L 2 alice-rico-mvp/
```

### **Step 2: Place Input Video**

```bash
# Copy your Google Whisk video to input folder
cp /path/to/your/whisk_video.mp4 input/base_video_static.mp4

# Verify video properties
python << EOF
import cv2
cap = cv2.VideoCapture('input/base_video_static.mp4')
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"Duration: {cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS):.2f}s")
cap.release()
EOF
```

### **Step 3: Configure Viseme Timestamps**

Create `config/viseme_config.json`:

```json
{
  "project_name": "Alice Avatar Phase 0",
  "version": "1.0",
  
  "video_source": "input/base_video_static.mp4",
  "fps": 30,
  "duration_seconds": 8.0,
  
  "viseme_timestamps": {
    "AA": null,
    "AE": null,
    "AH": null,
    "AO": null,
    "EH": null,
    "ER": null,
    "EY": null,
    "IH": null,
    "IY": null,
    "OW": null,
    "UH": null,
    "UW": null,
    "M": null,
    "F": null,
    "TH": null
  },
  
  "extraction_params": {
    "roi_padding": 25,
    "min_mouth_width": 100,
    "min_mouth_height": 60,
    "save_verification_images": true,
    "save_contact_sheet": true
  },
  
  "quality_thresholds": {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "min_face_area_ratio": 0.05
  }
}
```

**ACTION REQUIRED:** Watch your video and fill in timestamp values (in seconds) for each viseme. Example:

```json
"viseme_timestamps": {
  "AA": 0.5,   // When avatar says "father" - wide open mouth
  "AE": 1.2,   // When avatar says "cat" - medium open
  "AH": 1.9,   // When avatar says "cut"
  // ... etc
}
```

### **Step 4: Configure Blending Parameters**

Create `config/blending_config.json`:

```json
{
  "blending_mode": "poisson",
  "feather_percent": 0.25,
  "alignment_tolerance_pixels": 5,
  "use_gpu_acceleration": false,
  
  "poisson_params": {
    "clone_type": "normal",
    "blend_type": "mixed"
  },
  
  "alpha_blend_params": {
    "feather_width": 15,
    "feather_height": 10
  },
  
  "performance": {
    "max_workers": 4,
    "cache_visemes": true,
    "output_quality": "high"
  }
}
```

### **Step 5: Create Phoneme Mapping**

Create `config/phoneme_map.json`:

```json
{
  "phoneme_to_viseme_map": {
    "AA0": "AA", "AA1": "AA", "AA2": "AA",
    "AE0": "AE", "AE1": "AE", "AE2": "AE",
    "AH0": "AH", "AH1": "AH", "AH2": "AH",
    "AO0": "AO", "AO1": "AO", "AO2": "AO",
    "AW0": "AO", "AW1": "AO", "AW2": "AO",
    "AY0": "EY", "AY1": "EY", "AY2": "EY",
    "B": "M",
    "CH": "TH",
    "D": "TH",
    "DH": "TH",
    "EH0": "EH", "EH1": "EH", "EH2": "EH",
    "ER0": "ER", "ER1": "ER", "ER2": "ER",
    "EY0": "EY", "EY1": "EY", "EY2": "EY",
    "F": "F",
    "G": "AH",
    "HH": "AH",
    "IH0": "IH", "IH1": "IH", "IH2": "IH",
    "IY0": "IY", "IY1": "IY", "IY2": "IY",
    "JH": "TH",
    "K": "AH",
    "L": "TH",
    "M": "M",
    "N": "M",
    "NG": "M",
    "OW0": "OW", "OW1": "OW", "OW2": "OW",
    "OY0": "OW", "OY1": "OW", "OY2": "OW",
    "P": "M",
    "R": "ER",
    "S": "TH",
    "SH": "TH",
    "T": "TH",
    "TH": "TH",
    "UH0": "UH", "UH1": "UH", "UH2": "UH",
    "UW0": "UW", "UW1": "UW", "UW2": "UW",
    "V": "F",
    "W": "UW",
    "Y": "IY",
    "Z": "TH",
    "ZH": "TH"
  },
  
  "silence_phonemes": ["SIL", "SP", "spn"],
  
  "coarticulation_rules": {
    "enabled": false,
    "blend_duration_ms": 50
  }
}
```

---

## 🧪 **VERIFY INSTALLATION**

### **Test 1: Check Dependencies**

```bash
python << EOF
import sys
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"NumPy version: {np.__version__}")

# Test MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh()
print("✓ MediaPipe Face Mesh initialized")
face_mesh.close()

print("\n✅ All dependencies working correctly!")
EOF
```

### **Test 2: Verify Video Access**

```bash
python << EOF
import cv2

cap = cv2.VideoCapture('input/base_video_static.mp4')
if not cap.isOpened():
    print("❌ Error: Cannot open video file")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Error: Cannot read video frame")
    exit(1)

print(f"✓ Video loaded successfully")
print(f"  Frame shape: {frame.shape}")
cap.release()
EOF
```

### **Test 3: Face Detection Test**

```bash
python << EOF
import cv2
import mediapipe as mp

cap = cv2.VideoCapture('input/base_video_static.mp4')
ret, frame = cap.read()

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)

if results.multi_face_landmarks:
    print("✓ Face detected in first frame")
    landmarks = results.multi_face_landmarks[0]
    print(f"  Total landmarks: {len(landmarks.landmark)}")
else:
    print("⚠ Warning: No face detected - check video quality")

face_mesh.close()
cap.release()
EOF
```

---

## 🎯 **TIMESTAMP IDENTIFICATION WORKFLOW**

### **Method 1: Using VLC Media Player**

1. Open video in VLC: `vlc input/base_video_static.mp4`
2. Enable time display: `Tools → Preferences → Show settings: All → Video → On-screen display → Display time on video`
3. Play video and use `E` key to step forward frame-by-frame
4. When you see a clear viseme, note the timestamp
5. Record in `config/viseme_config.json`

### **Method 2: Using Python Script**

Create `tools/find_timestamps.py`:

```python
import cv2
import sys

def find_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("Controls:")
    print("  SPACE - Pause/Play")
    print("  → - Next frame")
    print("  ← - Previous frame")
    print("  M - Mark current timestamp")
    print("  Q - Quit")
    print()
    
    frame_num = 0
    paused = True
    marked_times = []
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
        
        timestamp = frame_num / fps
        
        # Display info
        info_frame = frame.copy()
        cv2.putText(info_frame, f"Time: {timestamp:.3f}s  Frame: {frame_num}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Find Timestamps', info_frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord(' '):  # Space - pause/play
            paused = not paused
        elif key == 83:  # Right arrow - next frame
            frame_num += 1
            paused = True
        elif key == 81:  # Left arrow - previous frame
            frame_num = max(0, frame_num - 1)
            paused = True
        elif key == ord('m'):  # Mark timestamp
            marked_times.append(timestamp)
            print(f"Marked: {timestamp:.3f}s")
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nMarked timestamps:")
    for t in marked_times:
        print(f"  {t:.3f}")

if __name__ == "__main__":
    find_timestamps('input/base_video_static.mp4')
```

Run: `python tools/find_timestamps.py`

---

## ✅ **PRE-FLIGHT CHECKLIST**

Before proceeding to viseme extraction:

- [ ] Python 3.9+ installed and verified
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Input video placed in `input/base_video_static.mp4`
- [ ] Video properties verified (resolution, FPS, duration)
- [ ] Face detection test passed
- [ ] Config files created in `config/` folder
- [ ] Viseme timestamps identified and filled in
- [ ] Project directory structure created
- [ ] All verification tests passed

---

## 🆘 **TROUBLESHOOTING**

### **Issue: MediaPipe installation fails**

```bash
# Try installing from source
pip install --no-binary mediapipe mediapipe
```

### **Issue: OpenCV cannot open video**

```bash
# Install additional codecs
pip install opencv-python-headless
# Or on Ubuntu:
sudo apt-get install ffmpeg
```

### **Issue: Face not detected**

- Check video has clear frontal face view
- Ensure adequate lighting in video
- Try lowering `min_detection_confidence` in config
- Verify video is not corrupted

### **Issue: Python version conflicts**

```bash
# Use specific Python version
python3.9 -m venv venv
```

---

## 📚 **NEXT STEPS**

Once setup is complete:

1. **Read:** `02-VISEME-EXTRACTION-SPEC.md`
2. **Run:** `python src/01_extract_visemes.py`
3. **Review:** Generated contact sheet in `output/visemes/`
4. **Proceed:** To RICo blending phase

---

**Setup complete? Continue to viseme extraction! →**
