# Alice Avatar - RICo Layer MVP
## Emotion-Aware Avatar with Real-Time Lip Sync

**Phase 0: Proof of Concept**  
**Goal:** Validate viseme layer blending technology for patent filing

---

## 🚀 QUICK START

### 1. **Setup Environment**
```bash
# Install Python 3.10 (recommended for MediaPipe)
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Place Your Video**
```bash
# Copy your 8-second Google Whisk video
cp /path/to/whisk_video.mp4 input/base_video_static.mp4
```

### 3. **Configure Timestamps**
```bash
# Watch video and identify when each phoneme appears
vlc input/base_video_static.mp4

# Edit config/viseme_config.json
# Fill in timestamps for each viseme (see template below)
```

### 4. **Extract Visemes**
```bash
python src/01_extract_visemes.py

# Review output
open output/visemes/contact_sheet.png
```

### 5. **Validate Results**
```bash
# Check extraction report
cat output/metadata/extraction_report.txt

# If quality is good (>85%), proceed to blending
# If quality is poor, adjust timestamps and re-run
```

---

## 📁 PROJECT STRUCTURE

```
alice-rico-mvp/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
│
├── docs/                        ← Full documentation
│   ├── 00-PROJECT-OVERVIEW.md
│   ├── 01-SETUP-GUIDE.md
│   ├── 02-VISEME-EXTRACTION-SPEC.md
│   └── 03-RICO-BLENDING-SPEC.md
│
├── input/                       ← Your videos go here
│   └── base_video_static.mp4    ← 8s Google Whisk video
│
├── output/                      ← Generated files
│   ├── visemes/                 ← 15 mouth images
│   │   ├── AA.png
│   │   ├── AE.png
│   │   ├── ...
│   │   ├── contact_sheet.png    ← Visual review
│   │   └── viseme_library.pkl   ← Runtime library
│   └── metadata/                ← Extraction reports
│
├── config/                      ← Configuration
│   ├── viseme_config.json       ← Timestamps go here
│   ├── blending_config.json
│   └── phoneme_map.json
│
└── src/                         ← Python scripts
    ├── 01_extract_visemes.py    ← Viseme extraction
    ├── 02_rico_blender.py       ← Layer blending (TODO)
    └── 03_integration_test.py   ← Full pipeline (TODO)
```

---

## ⚙️ CONFIGURATION TEMPLATE

### **config/viseme_config.json**

```json
{
  "project_name": "Alice Avatar Phase 0",
  "version": "1.0",
  
  "video_source": "input/base_video_static.mp4",
  "fps": 30,
  "duration_seconds": 8.0,
  
  "viseme_timestamps": {
    "AA": null,   ← Replace null with timestamp in seconds
    "AE": null,   ← Example: 1.2 (when avatar says "cat")
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

**HOW TO FILL IN TIMESTAMPS:**

1. Open video in VLC or similar player
2. Use frame-by-frame stepping (E key in VLC)
3. For each phoneme, find the frame where mouth shape is clearest
4. Note the timestamp (visible in player)
5. Enter timestamp in JSON config
6. Repeat for all 15 visemes

**Example filled config:**
```json
"viseme_timestamps": {
  "AA": 0.5,   // "father" - wide open
  "AE": 1.2,   // "cat" - medium spread
  "AH": 1.9,   // "cut" - neutral open
  "AO": 2.6,   // "caught" - rounded
  "EH": 3.3,   // "bet" - slight smile
  "ER": 4.0,   // "bird" - r-shape
  "EY": 4.7,   // "face" - spread smile
  "IH": 5.4,   // "bit" - small open
  "IY": 6.1,   // "beat" - wide smile
  "OW": 6.8,   // "go" - rounded forward
  "UH": 7.5,   // "book" - small round
  "UW": 8.2,   // "boot" - very rounded
  "M": 8.9,    // "mat" - lips closed
  "F": 9.6,    // "fan" - lip on teeth
  "TH": 10.3   // "thin" - tongue visible
}
```

---

## 📊 VISEME REFERENCE CHART

| Viseme | IPA | Example | Mouth Shape |
|--------|-----|---------|-------------|
| AA | /ɑː/ | f**a**ther | Wide open, jaw dropped |
| AE | /æ/ | c**a**t | Medium open, wide |
| AH | /ʌ/ | c**u**t | Medium open, neutral |
| AO | /ɔː/ | c**augh**t | Rounded, medium |
| EH | /ɛ/ | b**e**t | Slight smile, spread |
| ER | /ɜː/ | b**ir**d | Slightly open, r-shape |
| EY | /eɪ/ | f**a**ce | Spread smile |
| IH | /ɪ/ | b**i**t | Small open, spread |
| IY | /iː/ | b**ea**t | Wide smile, teeth |
| OW | /oʊ/ | g**o** | Rounded, forward |
| UH | /ʊ/ | b**oo**k | Small round |
| UW | /uː/ | b**oo**t | Very rounded |
| M | /m/b/p/ | **m**at | Lips closed |
| F | /f/v/ | **f**an | Lip on teeth |
| TH | /θ/ð/ | **th**in | Tongue visible |

---

## ✅ SUCCESS CRITERIA

**Extraction is successful when:**

- [ ] 13+ visemes extracted (87% success rate)
- [ ] Average quality score ≥ 85
- [ ] Contact sheet shows clear mouth shapes
- [ ] No critical visual issues
- [ ] Ready for RICo blending phase

**If extraction fails:**
- Adjust problematic timestamps ±0.2 seconds
- Check video has clear face visibility
- Verify lighting is adequate
- Re-run extraction

---

## 🐛 TROUBLESHOOTING

### **Problem: "Cannot open video"**
**Solution:**
- Verify file exists: `ls -l input/base_video_static.mp4`
- Check file format is MP4
- Try re-encoding: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### **Problem: "No face detected"**
**Solution:**
- Ensure face is clearly visible in video
- Check lighting is adequate
- Lower `min_detection_confidence` in config
- Verify video is not corrupted

### **Problem: "Some visemes have low quality"**
**Solution:**
- Try different timestamp (±0.1 to 0.5 seconds)
- Check if mouth is moving too fast at that timestamp
- Find clearer frame for that phoneme
- Re-extract specific visemes

### **Problem: "Motion blur detected"**
**Solution:**
- Viseme was extracted during mouth movement
- Find frame where mouth is at peak position
- Avoid transition frames
- Use frame with mouth stationary

---

## 📚 DOCUMENTATION

**Full documentation in `docs/` folder:**

1. **00-PROJECT-OVERVIEW.md** - System architecture, goals
2. **01-SETUP-GUIDE.md** - Detailed installation, environment setup
3. **02-VISEME-EXTRACTION-SPEC.md** - Technical extraction details
4. **03-RICO-BLENDING-SPEC.md** - Blending algorithm specification

**Read these for deep understanding of the system.**

---

## 🎯 NEXT STEPS

### **After successful extraction:**

1. **Review output**
   - Open `output/visemes/contact_sheet.png`
   - Verify all visemes look correct
   - Check `output/metadata/extraction_report.txt`

2. **Implement RICo blending** (Phase 2)
   - Use extracted visemes
   - Implement seamless compositing
   - Generate test videos

3. **Integrate TTS** (Phase 3)
   - Add text-to-speech engine
   - Extract phoneme timeline
   - Full pipeline test

4. **File patent** (Phase 4)
   - Document novel innovations
   - Prepare PPA filing
   - Secure IP before fundraising

---

## 📞 SUPPORT

**Common Issues:**
- Check `docs/01-SETUP-GUIDE.md` for installation problems
- Review extraction report for quality issues
- Verify config has all timestamps filled in

**For coding agent:**
- All specs are in `docs/` folder
- Follow implementation guides exactly
- Run tests after each component
- Document any deviations

---

## 🚀 QUICK COMMANDS

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Extract visemes
python src/01_extract_visemes.py

# View results
open output/visemes/contact_sheet.png
cat output/metadata/extraction_report.txt

# Check quality
grep "quality_score" output/metadata/extraction_metadata.json
```

---

## 📝 REQUIREMENTS.TXT

Create `requirements.txt` with:

```
opencv-python==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
Pillow==10.1.0
```

---

**Ready to extract visemes? Start with the config file! →**

**Questions? Read:** `docs/01-SETUP-GUIDE.md`
