# VISEME EXTRACTION SPECIFICATION
## Technical Implementation Guide

---

## 🎯 **OBJECTIVE**

Extract 15 clear mouth shape images (visemes) from the 8-second Google Whisk video, maintaining consistent positioning, lighting, and quality for seamless blending.

---

## 📊 **VISEME LIBRARY SPECIFICATION**

### **Complete Viseme Set:**

| Viseme ID | Phonemes | IPA | Example Words | Mouth Shape Description |
|-----------|----------|-----|---------------|------------------------|
| **AA** | /ɑː/ | AA, AO | f**a**ther, h**o**t | Wide open, jaw dropped, tongue low |
| **AE** | /æ/ | AE | c**a**t, b**a**t | Medium open, wide horizontal, tongue forward |
| **AH** | /ʌ/ | AH | c**u**t, b**u**s | Medium open, relaxed, neutral tongue |
| **AO** | /ɔː/ | AO, AW | c**augh**t, th**ough**t | Rounded lips, medium open |
| **EH** | /ɛ/ | EH | b**e**t, s**ai**d | Medium open, slight smile, spread |
| **ER** | /ɜː/ | ER | b**ir**d, h**er** | Slightly open, r-colored, lips neutral |
| **EY** | /eɪ/ | EY, AY | f**a**ce, d**ay** | Spread lips, slight smile, small opening |
| **IH** | /ɪ/ | IH | b**i**t, s**i**t | Small opening, lips spread slightly |
| **IY** | /iː/ | IY | b**ea**t, s**ee** | Wide smile, teeth may show, very spread |
| **OW** | /oʊ/ | OW, OY | g**o**, b**oa**t | Rounded, forward lips, medium opening |
| **UH** | /ʊ/ | UH | b**oo**k, g**oo**d | Small rounded opening, lips forward |
| **UW** | /uː/ | UW | b**oo**t, f**oo**d | Very rounded, forward lips, small opening |
| **M** | /m/b/p/ | M, B, P, N | **m**at, **b**at, **p**at | Lips pressed together, closed |
| **F** | /f/v/ | F, V | **f**an, **v**an | Lower lip on upper teeth |
| **TH** | /θ/ð/ | TH, DH, S, Z, T, D, L | **th**in, **th**at | Tongue visible between/behind teeth |

---

## 🔧 **EXTRACTION ALGORITHM**

### **Input Parameters:**
- Video file: `input/base_video_static.mp4`
- FPS: 30
- Duration: 8 seconds (240 frames)
- Timestamps: User-defined in `config/viseme_config.json`

### **Output Artifacts:**
- 15 PNG images: `output/visemes/{VISEME_ID}.png`
- Contact sheet: `output/visemes/contact_sheet.png`
- Verification images: `output/visemes/{VISEME_ID}_verification.png`
- Metadata: `output/metadata/extraction_metadata.json`
- Pickle library: `output/visemes/viseme_library.pkl`

---

## 📐 **ROI DETECTION METHODOLOGY**

### **MediaPipe Face Mesh Landmarks:**

```
Key Mouth Landmarks (468-point model):
- 61: Left mouth corner
- 291: Right mouth corner
- 0: Top lip center (cupid's bow)
- 17: Bottom lip center
- 39: Left top lip outer
- 269: Right top lip outer
- 13: Left bottom lip outer
- 14: Right bottom lip outer
```

### **ROI Calculation:**

```python
def calculate_mouth_roi(landmarks, frame_shape, padding=25):
    """
    Calculate consistent bounding box for mouth region
    
    Args:
        landmarks: MediaPipe face landmarks
        frame_shape: (height, width, channels)
        padding: Pixels to add around detected mouth
        
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    h, w = frame_shape[:2]
    
    # Extract mouth landmark coordinates
    mouth_indices = [61, 291, 0, 17, 39, 269, 13, 14]
    points = []
    
    for idx in mouth_indices:
        x = int(landmarks.landmark[idx].x * w)
        y = int(landmarks.landmark[idx].y * h)
        points.append((x, y))
    
    # Calculate bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    x_min = max(0, min(xs) - padding)
    y_min = max(0, min(ys) - padding)
    x_max = min(w, max(xs) + padding)
    y_max = min(h, max(ys) + padding)
    
    return (x_min, y_min, x_max, y_max)
```

---

## 🎨 **QUALITY VALIDATION**

### **Automatic Checks:**

```python
def validate_viseme_quality(viseme_image, metadata):
    """
    Validate extracted viseme meets quality standards
    
    Returns: (is_valid, quality_score, issues)
    """
    issues = []
    
    # Check 1: Minimum size
    h, w = viseme_image.shape[:2]
    if w < 100 or h < 60:
        issues.append(f"Too small: {w}x{h}")
    
    # Check 2: Motion blur detection
    laplacian = cv2.Laplacian(viseme_image, cv2.CV_64F)
    blur_score = laplacian.var()
    if blur_score < 100:
        issues.append(f"Motion blur detected: {blur_score:.1f}")
    
    # Check 3: Brightness consistency
    mean_brightness = viseme_image.mean()
    if mean_brightness < 40 or mean_brightness > 220:
        issues.append(f"Lighting issue: {mean_brightness:.1f}")
    
    # Check 4: Aspect ratio
    aspect_ratio = w / h
    if aspect_ratio < 1.2 or aspect_ratio > 2.5:
        issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
    
    # Calculate overall quality score
    quality_score = 100
    quality_score -= len(issues) * 20
    quality_score -= (100 - blur_score) if blur_score < 100 else 0
    
    is_valid = len(issues) == 0 and quality_score >= 70
    
    return is_valid, quality_score, issues
```

### **Manual Review Checklist:**

For each extracted viseme, verify:

- [ ] **Mouth shape clearly matches expected phoneme**
  - Compare against IPA reference chart
  - Check tongue position (if visible)
  - Verify lip rounding/spreading

- [ ] **No visual artifacts**
  - No motion blur
  - No compression artifacts
  - No partial occlusion

- [ ] **Consistent with other visemes**
  - Similar ROI size (±20%)
  - Same face angle
  - Same lighting direction

- [ ] **Adequate resolution**
  - Minimum 150x100 pixels
  - Clear edges, no pixelation
  - Sufficient detail for blending

---

## 🔄 **EXTRACTION WORKFLOW**

### **Phase 1: Preparation**

```
1. Load video: input/base_video_static.mp4
2. Validate video properties:
   - Resolution ≥ 720p
   - FPS = 30
   - Duration ≥ 8s
3. Initialize MediaPipe Face Mesh
4. Load config: config/viseme_config.json
```

### **Phase 2: Extraction Loop**

```
For each viseme in VISEME_PHONEMES:
  1. Get timestamp from config
  2. Seek to frame: frame_num = timestamp * fps
  3. Read frame
  4. Detect face landmarks
  5. Calculate mouth ROI
  6. Extract mouth region
  7. Validate quality
  8. Save files:
     - viseme image (PNG)
     - verification image (full frame with ROI box)
     - metadata (JSON)
```

### **Phase 3: Post-Processing**

```
1. Generate contact sheet (all visemes in grid)
2. Calculate quality statistics
3. Identify failed extractions
4. Save viseme library (pickle)
5. Generate extraction report
```

---

## 📦 **OUTPUT FILE FORMATS**

### **Individual Viseme Image (PNG):**

```
Filename: {VISEME_ID}.png
Format: PNG (lossless)
Color: RGB
Depth: 8-bit per channel
Dimensions: Variable (typically 150-250px width)
```

### **Verification Image (PNG):**

```
Filename: {VISEME_ID}_verification.png
Format: PNG
Content: Full frame with:
  - Green rectangle showing ROI
  - Phoneme label
  - Timestamp annotation
Purpose: Visual confirmation of extraction
```

### **Contact Sheet (PNG):**

```
Filename: contact_sheet.png
Layout: 5 columns × 3 rows grid
Cell size: max_viseme_size + 40px height for label
Purpose: Quick visual review of all visemes
```

### **Metadata (JSON):**

```json
{
  "extraction_timestamp": "2025-11-06T16:51:00Z",
  "video_source": "input/base_video_static.mp4",
  "video_properties": {
    "resolution": "1920x1080",
    "fps": 30,
    "duration_seconds": 8.0
  },
  "visemes": {
    "AA": {
      "timestamp": 0.5,
      "frame_number": 15,
      "roi_bounds": [450, 620, 680, 750],
      "image_size": [230, 130],
      "quality_score": 95.2,
      "is_valid": true,
      "issues": []
    },
    ...
  },
  "summary": {
    "total_visemes": 15,
    "successful": 15,
    "failed": 0,
    "average_quality": 92.3
  }
}
```

### **Pickle Library:**

```python
# Structure of viseme_library.pkl
{
    'AA': {
        'phoneme': 'AA',
        'image': np.ndarray,  # RGB image array
        'bounds': (x_min, y_min, x_max, y_max),
        'timestamp': 0.5,
        'frame_number': 15,
        'quality_score': 95.2
    },
    ...
}
```

---

## 🐛 **ERROR HANDLING**

### **Common Failure Modes:**

**1. Face Not Detected**
```python
if not results.multi_face_landmarks:
    # Retry with lower confidence
    # Try adjacent frames (±1, ±2)
    # Log warning and skip
```

**2. Poor Quality Viseme**
```python
if quality_score < 70:
    # Log warning
    # Try ±0.1s timestamp
    # Manual review required
```

**3. Timestamp Out of Bounds**
```python
if timestamp >= video_duration:
    # Log error
    # Skip viseme
    # Add to failed list
```

### **Recovery Strategies:**

```
If extraction fails for viseme X:
  1. Try frame at timestamp + 0.033s (next frame)
  2. Try frame at timestamp - 0.033s (previous frame)
  3. Try frame at timestamp + 0.100s
  4. Try frame at timestamp - 0.100s
  5. If all fail, mark as MANUAL_REVIEW_REQUIRED
  6. Continue with other visemes
  7. Generate report of failures
```

---

## 📊 **PERFORMANCE TARGETS**

### **Processing Speed:**
- **Single viseme extraction:** < 0.5 seconds
- **Complete library (15 visemes):** < 10 seconds
- **Contact sheet generation:** < 2 seconds
- **Total pipeline:** < 15 seconds

### **Quality Targets:**
- **Successful extraction rate:** ≥ 90% (13/15 visemes)
- **Average quality score:** ≥ 85/100
- **ROI consistency:** ≤ 10% size variance
- **No manual intervention required:** Best case

---

## 🔍 **VALIDATION CHECKLIST**

After extraction, verify:

### **File System:**
- [ ] 15 viseme PNG files exist in `output/visemes/`
- [ ] 15 verification PNG files exist
- [ ] Contact sheet generated
- [ ] Metadata JSON created
- [ ] Pickle library saved

### **Quality:**
- [ ] All visemes have quality_score ≥ 70
- [ ] No critical issues reported
- [ ] Contact sheet shows clear mouth shapes
- [ ] Viseme sizes are consistent (±20%)

### **Visual Inspection:**
- [ ] Open contact sheet
- [ ] Verify each viseme matches expected mouth shape
- [ ] Check for artifacts, blur, or misalignment
- [ ] Confirm lighting consistency

---

## 🚨 **TROUBLESHOOTING GUIDE**

### **Problem: Multiple faces detected**
**Solution:**
- Use `max_num_faces=1` in MediaPipe config
- Ensure video has single person centered

### **Problem: Mouth ROI drifts between frames**
**Solution:**
- Verify face is stable in video
- Use stricter face tracking parameters
- Consider video stabilization pre-process

### **Problem: Some visemes are blurry**
**Solution:**
- Check if mouth is moving too fast at timestamp
- Try different timestamp ±0.2s
- Use frame with mouth at peak position

### **Problem: Inconsistent ROI sizes**
**Solution:**
- Adjust padding parameter
- Normalize ROI to fixed aspect ratio
- Use median size as reference

---

## 📝 **EXTRACTION REPORT TEMPLATE**

```
=== VISEME EXTRACTION REPORT ===
Date: 2025-11-06 16:51:00
Video: input/base_video_static.mp4
Duration: 8.00s | FPS: 30 | Resolution: 1920x1080

=== EXTRACTION RESULTS ===
Total visemes attempted: 15
Successful: 14 (93.3%)
Failed: 1 (6.7%)

=== QUALITY METRICS ===
Average quality score: 87.4
Minimum quality: 72.1 (EH)
Maximum quality: 96.8 (IY)

=== FAILED EXTRACTIONS ===
- TH: No face detected at timestamp 10.3s
  → Recommendation: Try timestamp 10.2s or 10.4s

=== WARNINGS ===
- AH: Quality score 74.2 (below 80 threshold)
  → Recommendation: Manual review advised
- ER: ROI size 15% smaller than average
  → Recommendation: Check if acceptable

=== NEXT STEPS ===
1. Review contact sheet: output/visemes/contact_sheet.png
2. Address failed extractions manually
3. Re-run for low quality visemes if needed
4. Proceed to RICo blending when satisfied
```

---

## ✅ **SUCCESS CRITERIA**

Extraction is complete when:

- [ ] ≥ 13 visemes extracted successfully (87%)
- [ ] All extracted visemes have quality_score ≥ 70
- [ ] Contact sheet reviewed and approved
- [ ] No critical visual issues identified
- [ ] Metadata validation passed
- [ ] Ready for RICo blending phase

---

**Extraction complete? Proceed to:** `03-RICO-BLENDING-SPEC.md`
