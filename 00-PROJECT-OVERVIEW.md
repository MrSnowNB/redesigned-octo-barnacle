# ALICE AVATAR - RICO LAYER PROJECT
## Project Overview & Architecture

**Version:** 1.0  
**Status:** Phase 0 - Proof of Concept  
**Goal:** Validate viseme layer blending for emotion-aware avatar lip sync

---

## 🎯 **OBJECTIVE**

Prove that **viseme layer blending** produces visually convincing lip sync when overlaid on emotion-appropriate base video using the RICo (Real-time Image Composition) system.

---

## 🏗️ **SYSTEM ARCHITECTURE**

```
┌─────────────────────────────────────────────────────┐
│              INPUT: Static Base Video                │
│          (Neutral expression, closed mouth)          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────────┐
         │  VISEME EXTRACTION      │
         │  From 8s reference video │
         │  → 15 mouth shape images │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │  TTS GENERATION         │
         │  Text → Phoneme timeline │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │  RICO LAYER BLENDING    │
         │  1. Detect mouth ROI    │
         │  2. Align viseme layer  │
         │  3. Seamless blend      │
         └──────────┬────────────────┘
                    │
                    ▼
         ┌─────────────────────────┐
         │  OUTPUT: Synced Video   │
         │  Base + animated mouth  │
         └─────────────────────────┘
```

---

## 📁 **PROJECT STRUCTURE**

```
alice-rico-mvp/
├── docs/
│   ├── 00-PROJECT-OVERVIEW.md          (this file)
│   ├── 01-SETUP-GUIDE.md               (environment setup)
│   ├── 02-VISEME-EXTRACTION-SPEC.md    (extraction process)
│   ├── 03-RICO-BLENDING-SPEC.md        (blending algorithm)
│   ├── 04-API-REFERENCE.md             (code documentation)
│   └── 05-QUALITY-CHECKLIST.md         (validation criteria)
│
├── input/
│   ├── base_video_static.mp4           (YOUR 8s video from Whisk)
│   └── test_text.txt                   (test sentences for TTS)
│
├── output/
│   ├── visemes/                        (extracted mouth images)
│   ├── metadata/                       (extraction metadata)
│   ├── test_videos/                    (generated test outputs)
│   └── logs/                           (processing logs)
│
├── src/
│   ├── 01_extract_visemes.py           (viseme extraction)
│   ├── 02_rico_blender.py              (layer blending engine)
│   ├── 03_integration_test.py          (full pipeline test)
│   └── utils/
│       ├── face_detection.py           (MediaPipe wrapper)
│       ├── tts_wrapper.py              (TTS phoneme extraction)
│       └── video_utils.py              (video I/O helpers)
│
├── config/
│   ├── viseme_config.json              (viseme parameters)
│   ├── blending_config.json            (blending parameters)
│   └── phoneme_map.json                (phoneme → viseme mapping)
│
├── requirements.txt                     (Python dependencies)
├── README.md                           (Quick start guide)
└── run_pipeline.py                     (main execution script)
```

---

## 🔑 **KEY COMPONENTS**

### **1. Viseme Extraction (`01_extract_visemes.py`)**

**Purpose:** Extract 15 mouth shape images from 8-second reference video

**Input:**
- `input/base_video_static.mp4` (8s video from Google Whisk)
- `config/viseme_config.json` (manual timestamps)

**Output:**
- 15 PNG images in `output/visemes/`
- `viseme_library.pkl` (processed data)
- `extraction_metadata.json` (quality metrics)

**Process:**
1. Load video and detect face landmarks
2. For each viseme timestamp, extract mouth ROI
3. Save mouth region image
4. Generate verification images
5. Create contact sheet for visual review

---

### **2. RICo Blender (`02_rico_blender.py`)**

**Purpose:** Blend viseme layer onto base video with seamless compositing

**Input:**
- Base video (looping or static)
- Viseme library (15 images)
- Phoneme timeline (from TTS)

**Output:**
- Synced video with lip movements
- Frame-by-frame blend quality metrics

**Process:**
1. Detect mouth ROI in base frame
2. For each phoneme in timeline:
   - Select matching viseme
   - Align to current frame mouth position
   - Apply Poisson blending
   - Composite onto frame
3. Export final video

---

### **3. Integration Test (`03_integration_test.py`)**

**Purpose:** Validate complete pipeline with test sentences

**Test Cases:**
- Short phrase (3 seconds)
- Medium sentence (5 seconds)
- Long paragraph (10+ seconds)

**Success Criteria:**
- No visible seams in blending
- Lip sync matches TTS timing (±50ms)
- Maintains base video quality
- Processes at >20fps

---

## 📊 **TECHNICAL SPECIFICATIONS**

### **Video Format:**
- **Resolution:** 1920x1080 (1080p)
- **Frame Rate:** 30fps
- **Codec:** H.264
- **Format:** MP4

### **Viseme Library:**
- **Count:** 15 phonemes
- **Format:** PNG (lossless)
- **Size:** ~150-200px width each
- **Color Space:** RGB

### **Processing Requirements:**
- **Python:** 3.9+
- **OpenCV:** 4.8+
- **MediaPipe:** 0.10+
- **GPU:** Optional (CPU fallback available)

---

## 🎯 **SUCCESS METRICS**

### **Phase 0 Validation:**

1. **Visual Quality:**
   - [ ] No visible seams at mouth edges
   - [ ] Smooth transitions between phonemes
   - [ ] Maintains base video aesthetic

2. **Synchronization:**
   - [ ] Lip movements match audio within 50ms
   - [ ] Natural coarticulation between phonemes
   - [ ] Appropriate viseme duration

3. **Technical Performance:**
   - [ ] Processing speed >20fps on target hardware
   - [ ] Stable face tracking (no drift)
   - [ ] Consistent blending quality across video

4. **User Acceptance:**
   - [ ] 3+ testers prefer RICo version over static
   - [ ] Described as "natural" or "realistic"
   - [ ] Passes "uncanny valley" threshold

---

## 🚀 **EXECUTION WORKFLOW**

### **Step 1: Setup Environment**
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, mediapipe; print('Ready!')"
```

### **Step 2: Place Input Video**
```bash
# Copy your Google Whisk video
cp /path/to/whisk_video.mp4 input/base_video_static.mp4

# Verify video specs
python -c "import cv2; cap=cv2.VideoCapture('input/base_video_static.mp4'); \
print(f'Resolution: {int(cap.get(3))}x{int(cap.get(4))}'); \
print(f'FPS: {cap.get(5)}'); print(f'Duration: {cap.get(7)/cap.get(5):.2f}s')"
```

### **Step 3: Configure Viseme Timestamps**
```bash
# Watch video and mark timestamps
vlc input/base_video_static.mp4

# Edit config/viseme_config.json with timestamps
nano config/viseme_config.json
```

### **Step 4: Extract Visemes**
```bash
# Run extraction
python src/01_extract_visemes.py

# Review contact sheet
open output/visemes/contact_sheet.png
```

### **Step 5: Test RICo Blending**
```bash
# Run test with sample phrase
python src/03_integration_test.py --text "Hello, how are you?"

# View output
open output/test_videos/test_output_001.mp4
```

### **Step 6: Validation**
```bash
# Run quality checks
python src/utils/quality_validator.py

# Generate report
python src/utils/generate_report.py
```

---

## 📝 **CONFIGURATION FILES**

### **viseme_config.json**
```json
{
  "video_source": "input/base_video_static.mp4",
  "fps": 30,
  "timestamps": {
    "AA": 0.5,
    "AE": 1.0,
    "AH": 1.5,
    ...
  },
  "roi_padding": 25,
  "quality_threshold": 0.8
}
```

### **blending_config.json**
```json
{
  "blend_mode": "poisson",
  "feather_percent": 0.25,
  "alignment_tolerance": 5,
  "use_gpu": false
}
```

---

## 🔬 **TECHNICAL INNOVATIONS**

### **Novel Aspects for Patent:**

1. **Person-Specific Viseme Library**
   - Extracted from same source video
   - Maintains visual consistency
   - Emotion-aware (future expansion)

2. **Semantic Region Blending**
   - Blends mouth as semantic unit (not pixels)
   - Preserves surrounding facial features
   - Adaptive alignment based on face tracking

3. **Real-time Layer Composition**
   - Separates base emotion from articulation
   - Client-side reconstruction capability
   - Bandwidth-efficient streaming (future)

---

## 📞 **SUPPORT & TROUBLESHOOTING**

### **Common Issues:**

**Issue:** Face not detected in video
- **Fix:** Ensure face is clearly visible, good lighting
- **Fix:** Adjust `min_detection_confidence` in config

**Issue:** Viseme extraction fails for some phonemes
- **Fix:** Verify timestamps are accurate
- **Fix:** Check mouth is clearly visible at timestamp
- **Fix:** Try adjacent frames (±0.03s)

**Issue:** Blending shows visible seams
- **Fix:** Increase `feather_percent` value
- **Fix:** Try `poisson` blend mode instead of `alpha`
- **Fix:** Verify ROI alignment is stable

**Issue:** Processing too slow
- **Fix:** Enable GPU acceleration
- **Fix:** Reduce output resolution temporarily
- **Fix:** Use `alpha` blending instead of `poisson`

---

## 📅 **DEVELOPMENT PHASES**

### **Phase 0: Proof of Concept** ← YOU ARE HERE
- Single static video
- 15 viseme extraction
- Basic RICo blending
- Manual quality validation

### **Phase 1: Single Emotion System**
- One emotion base video (6s loop)
- Automated quality checks
- TTS integration
- Performance optimization

### **Phase 2: Multi-Emotion System**
- 7 emotion videos
- 7 × 15 viseme libraries
- Emotion detection
- Dynamic emotion selection

### **Phase 3: Production System**
- Real-time streaming
- Cloud deployment
- API development
- Patent filing

---

## 🎓 **LEARNING RESOURCES**

### **Key Technologies:**

**MediaPipe Face Mesh:**
- https://google.github.io/mediapipe/solutions/face_mesh.html
- 468 facial landmarks
- Real-time performance

**OpenCV Seamless Cloning:**
- https://docs.opencv.org/4.x/df/da0/group__photo__clone.html
- Poisson image editing
- Gradient-domain blending

**Phoneme-Viseme Mapping:**
- https://en.wikipedia.org/wiki/Viseme
- International Phonetic Alphabet (IPA)
- Coarticulation effects

---

## ✅ **NEXT STEPS**

1. **Read all documentation** in `docs/` folder
2. **Set up environment** per `01-SETUP-GUIDE.md`
3. **Place your video** in `input/` folder
4. **Configure timestamps** in `config/viseme_config.json`
5. **Run extraction** using `01_extract_visemes.py`
6. **Review visemes** in contact sheet
7. **Test blending** with `03_integration_test.py`
8. **Validate results** against success criteria

---

**Ready to build? Start with `01-SETUP-GUIDE.md`**
