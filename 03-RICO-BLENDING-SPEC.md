# RICO BLENDING SPECIFICATION
## Seamless Viseme Layer Compositing

---

## 🎯 **OBJECTIVE**

Blend extracted viseme images onto base video frames with imperceptible seams, creating photorealistic lip-synced avatar speech while maintaining emotional authenticity.

---

## 🏗️ **BLENDING PIPELINE ARCHITECTURE**

```
INPUT:
├── Base video frame (static or looping)
├── Viseme library (15 mouth images)
└── Phoneme timeline (from TTS)

PROCESSING:
│
├─> 1. FACE TRACKING
│   ├── Detect facial landmarks
│   ├── Calculate current mouth ROI
│   └── Track head movement frame-to-frame
│
├─> 2. VISEME SELECTION
│   ├── Match current phoneme to viseme
│   ├── Handle coarticulation (optional)
│   └── Load corresponding mouth image
│
├─> 3. ALIGNMENT
│   ├── Calculate transformation matrix
│   │   ├── Translation (if mouth moved)
│   │   ├── Rotation (if head tilted)
│   │   └── Scale (if zoom changed)
│   └── Warp viseme to match current frame
│
├─> 4. SEAMLESS COMPOSITING
│   ├── Create feathered blend mask
│   ├── Apply Poisson blending
│   └── Composite onto base frame
│
OUTPUT:
└── Synced video with natural lip movements
```

---

## 🎨 **BLENDING MODES**

### **Mode 1: Alpha Feathering (Fast)**

**Use Case:** Real-time preview, testing  
**Speed:** ~2ms per frame  
**Quality:** Good (90% of cases)

```python
def alpha_feather_blend(base_frame, viseme_img, roi_bounds, feather_percent=0.25):
    """
    Fast alpha blending with feathered edges
    
    Args:
        base_frame: Full video frame
        viseme_img: Extracted mouth region
        roi_bounds: (x_min, y_min, x_max, y_max)
        feather_percent: Edge feathering amount (0.0-0.5)
    
    Returns:
        Composited frame with blended mouth
    """
    x_min, y_min, x_max, y_max = roi_bounds
    h, w = y_max - y_min, x_max - x_min
    
    # Resize viseme to ROI size
    viseme_resized = cv2.resize(viseme_img, (w, h))
    
    # Create feathered alpha mask
    mask = create_feather_mask(w, h, feather_percent)
    
    # Extract base ROI
    base_roi = base_frame[y_min:y_max, x_min:x_max]
    
    # Alpha blend
    blended_roi = (base_roi * (1 - mask) + viseme_resized * mask).astype(np.uint8)
    
    # Copy back to frame
    result = base_frame.copy()
    result[y_min:y_max, x_min:x_max] = blended_roi
    
    return result

def create_feather_mask(width, height, feather_percent):
    """Create gradient mask for smooth edge transitions"""
    mask = np.ones((height, width), dtype=np.float32)
    
    feather_w = int(width * feather_percent)
    feather_h = int(height * feather_percent)
    
    # Feather all edges
    for i in range(feather_h):
        alpha = i / feather_h
        mask[i, :] *= alpha  # Top
        mask[height - 1 - i, :] *= alpha  # Bottom
    
    for i in range(feather_w):
        alpha = i / feather_w
        mask[:, i] *= alpha  # Left
        mask[:, width - 1 - i] *= alpha  # Right
    
    return mask[:, :, np.newaxis]  # Add channel dimension
```

---

### **Mode 2: Poisson Blending (High Quality)**

**Use Case:** Final output, investor demos  
**Speed:** ~50-100ms per frame  
**Quality:** Excellent (photorealistic)

```python
def poisson_blend(base_frame, viseme_img, roi_bounds):
    """
    Seamless Poisson blending using gradient-domain compositing
    
    Uses OpenCV's seamlessClone for imperceptible seams
    
    Args:
        base_frame: Full video frame
        viseme_img: Extracted mouth region
        roi_bounds: (x_min, y_min, x_max, y_max)
    
    Returns:
        Composited frame with seamless mouth blend
    """
    x_min, y_min, x_max, y_max = roi_bounds
    h, w = y_max - y_min, x_max - x_min
    
    # Resize viseme
    viseme_resized = cv2.resize(viseme_img, (w, h))
    
    # Create mask (white ellipse for mouth region)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 2 - 10, h // 2 - 10)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # Feather mask edges
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Calculate center point for Poisson
    center_x = x_min + w // 2
    center_y = y_min + h // 2
    
    # Seamless clone
    result = cv2.seamlessClone(
        viseme_resized,
        base_frame,
        mask,
        (center_x, center_y),
        cv2.NORMAL_CLONE
    )
    
    return result
```

---

## 🎯 **ALIGNMENT ALGORITHM**

### **Challenge:**
Face may have slight movement between base video and viseme extraction, causing misalignment.

### **Solution: Landmark-Based Transform**

```python
class MouthAligner:
    """Ensures pixel-perfect alignment of viseme layer"""
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_tracking_confidence=0.7
        )
        self.base_roi_cache = None
    
    def detect_mouth_params(self, frame):
        """
        Detect mouth position, rotation, and scale
        
        Returns: {
            'center': (x, y),
            'angle': rotation_degrees,
            'scale': mouth_width_pixels,
            'bounds': (x_min, y_min, x_max, y_max)
        }
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Key mouth landmarks
        left_corner = self._get_point(landmarks, 61, w, h)
        right_corner = self._get_point(landmarks, 291, w, h)
        top_lip = self._get_point(landmarks, 0, w, h)
        bottom_lip = self._get_point(landmarks, 17, w, h)
        
        # Calculate parameters
        center_x = (left_corner[0] + right_corner[0]) // 2
        center_y = (top_lip[1] + bottom_lip[1]) // 2
        
        # Rotation angle
        dx = right_corner[0] - left_corner[0]
        dy = right_corner[1] - left_corner[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Scale (mouth width)
        scale = np.sqrt(dx**2 + dy**2)
        
        # Bounding box
        padding = 25
        x_coords = [left_corner[0], right_corner[0]]
        y_coords = [top_lip[1], bottom_lip[1]]
        
        bounds = (
            max(0, min(x_coords) - padding),
            max(0, min(y_coords) - padding),
            min(w, max(x_coords) + padding),
            min(h, max(y_coords) + padding)
        )
        
        return {
            'center': (center_x, center_y),
            'angle': angle,
            'scale': scale,
            'bounds': bounds
        }
    
    def align_viseme(self, viseme_img, base_params, target_params):
        """
        Transform viseme to match target frame geometry
        
        Args:
            viseme_img: Original extracted viseme
            base_params: Mouth params when viseme was extracted
            target_params: Mouth params in current frame
        
        Returns:
            Aligned viseme image
        """
        # Calculate transformation
        scale_factor = target_params['scale'] / base_params['scale']
        angle_diff = target_params['angle'] - base_params['angle']
        
        # Translation vector
        dx = target_params['center'][0] - base_params['center'][0]
        dy = target_params['center'][1] - base_params['center'][1]
        
        # Build affine transformation matrix
        center = (viseme_img.shape[1] // 2, viseme_img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle_diff, scale_factor)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # Apply transformation
        h, w = viseme_img.shape[:2]
        aligned = cv2.warpAffine(viseme_img, M, (w, h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        return aligned
    
    def _get_point(self, landmarks, idx, w, h):
        """Extract landmark point coordinates"""
        return (
            int(landmarks.landmark[idx].x * w),
            int(landmarks.landmark[idx].y * h)
        )
```

---

## 🎬 **PHONEME TIMELINE PROCESSING**

### **Input Format (from TTS):**

```json
{
  "text": "Hello, how are you?",
  "duration": 2.5,
  "phonemes": [
    {"phoneme": "HH", "start": 0.0, "duration": 0.08},
    {"phoneme": "AH", "start": 0.08, "duration": 0.12},
    {"phoneme": "L", "start": 0.20, "duration": 0.10},
    {"phoneme": "OW", "start": 0.30, "duration": 0.15},
    {"phoneme": "SIL", "start": 0.45, "duration": 0.10},
    ...
  ]
}
```

### **Viseme Mapping:**

```python
def map_phoneme_to_viseme(phoneme, phoneme_map):
    """
    Map TTS phoneme to viseme ID
    
    Args:
        phoneme: ARPAbet phoneme (e.g., 'AH0', 'T', 'HH')
        phoneme_map: Loaded from config/phoneme_map.json
    
    Returns:
        Viseme ID (e.g., 'AH', 'TH', 'M')
    """
    # Remove stress markers (0, 1, 2)
    phoneme_base = ''.join(c for c in phoneme if c.isalpha())
    
    # Check silence
    if phoneme in phoneme_map['silence_phonemes']:
        return None  # Use base frame mouth
    
    # Map to viseme
    return phoneme_map['phoneme_to_viseme_map'].get(phoneme_base, 'AH')
```

---

## ⚡ **COARTICULATION (OPTIONAL)**

### **Problem:**
Abrupt viseme changes look robotic.

### **Solution:**
Blend between consecutive visemes during transitions.

```python
def apply_coarticulation(prev_viseme, next_viseme, progress):
    """
    Smooth transition between phonemes
    
    Args:
        prev_viseme: Previous mouth image
        next_viseme: Next mouth image
        progress: 0.0 (fully prev) to 1.0 (fully next)
    
    Returns:
        Blended intermediate mouth image
    """
    # Ensure same size
    h, w = prev_viseme.shape[:2]
    next_resized = cv2.resize(next_viseme, (w, h))
    
    # Crossfade
    alpha = progress
    blended = cv2.addWeighted(prev_viseme, 1 - alpha, next_resized, alpha, 0)
    
    return blended
```

**When to use:**
- Transition duration: 30-50ms (1-2 frames)
- Only between similar visemes (same openness category)
- Not for silence → speech transitions

---

## 🎥 **VIDEO GENERATION PIPELINE**

```python
def generate_synced_video(base_video_path, viseme_library, phoneme_timeline, 
                         output_path, blend_mode='poisson'):
    """
    Main pipeline: Generate lip-synced video
    
    Args:
        base_video_path: Path to base video (static or looping)
        viseme_library: Dict of {viseme_id: image}
        phoneme_timeline: List of phoneme events with timing
        output_path: Where to save result
        blend_mode: 'alpha' or 'poisson'
    
    Returns:
        Path to generated video
    """
    # Initialize
    cap = cv2.VideoCapture(base_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize aligner
    aligner = MouthAligner()
    
    # Get baseline mouth parameters (first frame)
    ret, first_frame = cap.read()
    base_params = aligner.detect_mouth_params(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
    
    # Process frames
    frame_num = 0
    current_time = 0.0
    frame_duration = 1.0 / fps
    
    # Build phoneme lookup for fast access
    phoneme_events = sorted(phoneme_timeline, key=lambda x: x['start'])
    
    while True:
        ret, base_frame = cap.read()
        if not ret:
            # Loop base video if needed
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, base_frame = cap.read()
            if not ret:
                break
        
        # Find current phoneme
        current_phoneme = None
        for event in phoneme_events:
            if event['start'] <= current_time < event['start'] + event['duration']:
                current_phoneme = event['phoneme']
                break
        
        # Generate frame
        if current_phoneme and current_phoneme in viseme_library:
            # Get viseme
            viseme_img = viseme_library[current_phoneme]['image']
            
            # Detect current frame mouth params
            target_params = aligner.detect_mouth_params(base_frame)
            
            if target_params:
                # Align viseme
                aligned_viseme = aligner.align_viseme(
                    viseme_img, base_params, target_params
                )
                
                # Blend
                if blend_mode == 'poisson':
                    synced_frame = poisson_blend(
                        base_frame, aligned_viseme, target_params['bounds']
                    )
                else:
                    synced_frame = alpha_feather_blend(
                        base_frame, aligned_viseme, target_params['bounds']
                    )
                
                out.write(synced_frame)
            else:
                # No face detected, use base frame
                out.write(base_frame)
        else:
            # Silence or no viseme, use base frame
            out.write(base_frame)
        
        frame_num += 1
        current_time += frame_duration
        
        # Stop when timeline complete
        if current_time >= phoneme_timeline[-1]['start'] + phoneme_timeline[-1]['duration']:
            break
    
    cap.release()
    out.release()
    
    return output_path
```

---

## 📊 **QUALITY METRICS**

### **Automatic Validation:**

```python
def validate_blend_quality(original_frame, blended_frame, roi_bounds):
    """
    Measure blending quality
    
    Returns: {
        'seam_visibility': 0-100 (lower is better),
        'color_continuity': 0-100 (higher is better),
        'sharpness': 0-100 (higher is better)
    }
    """
    x_min, y_min, x_max, y_max = roi_bounds
    
    # Extract ROI and surrounding area
    pad = 10
    roi_expanded = blended_frame[
        y_min-pad:y_max+pad, 
        x_min-pad:x_max+pad
    ]
    
    # 1. Seam visibility (gradient discontinuity at edges)
    edges = cv2.Canny(roi_expanded, 100, 200)
    seam_score = edges[pad:-pad, :pad].sum() + edges[pad:-pad, -pad:].sum()
    seam_score += edges[:pad, pad:-pad].sum() + edges[-pad:, pad:-pad].sum()
    seam_visibility = min(100, seam_score / 100)
    
    # 2. Color continuity
    roi_mean = roi_expanded[pad:-pad, pad:-pad].mean(axis=(0, 1))
    surround_mean = np.concatenate([
        roi_expanded[:pad, :].flatten(),
        roi_expanded[-pad:, :].flatten(),
        roi_expanded[:, :pad].flatten(),
        roi_expanded[:, -pad:].flatten()
    ]).reshape(-1, 3).mean(axis=0)
    color_diff = np.linalg.norm(roi_mean - surround_mean)
    color_continuity = max(0, 100 - color_diff)
    
    # 3. Sharpness
    laplacian = cv2.Laplacian(roi_expanded, cv2.CV_64F)
    sharpness = min(100, laplacian.var() / 10)
    
    return {
        'seam_visibility': seam_visibility,
        'color_continuity': color_continuity,
        'sharpness': sharpness
    }
```

---

## ✅ **SUCCESS CRITERIA**

Blending is successful when:

- [ ] **No visible seams** at mouth edges (seam_visibility < 20)
- [ ] **Color continuity** maintained (color_continuity > 80)
- [ ] **Natural transitions** between phonemes
- [ ] **Sync accuracy** within ±50ms of audio
- [ ] **Processing speed** adequate for use case:
  - Real-time preview: >15fps
  - Final render: >5fps acceptable
- [ ] **Visual quality** passes manual review

---

**Blending complete? Proceed to:** `03_integration_test.py`
