"""
01_extract_visemes.py

Viseme Extraction Script - Alice Avatar RICo Layer
Extracts 15 mouth shape images from 8-second reference video

Usage:
    python src/01_extract_visemes.py
    
Requirements:
    - Input video: input/base_video_static.mp4
    - Config: config/viseme_config.json with timestamps filled in
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

@dataclass
class VisemeData:
    """Container for extracted viseme information"""
    phoneme: str
    image: np.ndarray
    bounds: Tuple[int, int, int, int]
    timestamp: float
    frame_number: int
    quality_score: float
    is_valid: bool
    issues: list

class VisemeExtractor:
    """Extract viseme library from reference video"""
    
    VISEME_PHONEMES = [
        'AA', 'AE', 'AH', 'AO', 'EH', 'ER', 'EY',
        'IH', 'IY', 'OW', 'UH', 'UW', 'M', 'F', 'TH'
    ]
    
    def __init__(self, config_path: str = 'config/viseme_config.json'):
        """Initialize extractor with configuration"""
        # Load config
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Setup paths
        self.video_path = self.config['video_source']
        self.output_dir = Path('output/visemes')
        self.metadata_dir = Path('output/metadata')
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config['quality_thresholds']['min_detection_confidence'],
            min_tracking_confidence=self.config['quality_thresholds']['min_tracking_confidence']
        )
        
        print(f"\n{'='*70}")
        print(f"VISEME EXTRACTION - Alice Avatar RICo Layer")
        print(f"{'='*70}")
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Duration: {self.total_frames/self.fps:.2f}s")
        print(f"{'='*70}\n")
    
    def get_mouth_roi_bounds(self, landmarks, frame_shape: Tuple) -> Tuple[int, int, int, int]:
        """
        Calculate consistent mouth bounding box from facial landmarks
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: (height, width, channels)
            
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        h, w = frame_shape[:2]
        padding = self.config['extraction_params']['roi_padding']
        
        # Key mouth landmarks (outer lips)
        mouth_indices = [61, 291, 0, 17, 39, 269, 13, 14]
        
        # Extract coordinates
        mouth_points = []
        for idx in mouth_indices:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            mouth_points.append((x, y))
        
        # Calculate bounding box
        xs = [p[0] for p in mouth_points]
        ys = [p[1] for p in mouth_points]
        
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(w, max(xs) + padding)
        y_max = min(h, max(ys) + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_frame_at_timestamp(self, timestamp: float) -> Tuple[np.ndarray, int]:
        """
        Extract frame at specific timestamp
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            (frame, frame_number)
        """
        frame_num = int(timestamp * self.fps)
        
        if frame_num >= self.total_frames:
            raise ValueError(f"Timestamp {timestamp}s exceeds video duration")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp}s")
        
        return frame, frame_num
    
    def validate_viseme_quality(self, viseme_image: np.ndarray) -> Tuple[bool, float, list]:
        """
        Validate extracted viseme meets quality standards
        
        Returns:
            (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 100.0
        
        # Check 1: Minimum size
        h, w = viseme_image.shape[:2]
        min_w = self.config['extraction_params']['min_mouth_width']
        min_h = self.config['extraction_params']['min_mouth_height']
        
        if w < min_w or h < min_h:
            issues.append(f"Size too small: {w}x{h} (min {min_w}x{min_h})")
            quality_score -= 30
        
        # Check 2: Motion blur detection
        gray = cv2.cvtColor(viseme_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        if blur_score < 100:
            issues.append(f"Motion blur detected: {blur_score:.1f}")
            quality_score -= 20
        
        # Check 3: Brightness
        mean_brightness = viseme_image.mean()
        if mean_brightness < 40 or mean_brightness > 220:
            issues.append(f"Lighting issue: brightness={mean_brightness:.1f}")
            quality_score -= 15
        
        # Check 4: Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1.2 or aspect_ratio > 2.5:
            issues.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
            quality_score -= 10
        
        quality_score = max(0, quality_score)
        is_valid = quality_score >= 70 and len(issues) == 0
        
        return is_valid, quality_score, issues
    
    def extract_viseme(self, phoneme: str, timestamp: float) -> Optional[VisemeData]:
        """
        Extract single viseme at specified timestamp
        
        Args:
            phoneme: Phoneme code (e.g., 'AA', 'EH')
            timestamp: Time in seconds
            
        Returns:
            VisemeData object or None if extraction fails
        """
        try:
            # Get frame
            frame, frame_num = self.extract_frame_at_timestamp(timestamp)
            
            # Detect face landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                print(f"  ✗ {phoneme}: No face detected at {timestamp:.2f}s")
                return None
            
            landmarks = results.multi_face_landmarks[0]
            
            # Get mouth ROI
            x_min, y_min, x_max, y_max = self.get_mouth_roi_bounds(landmarks, frame.shape)
            mouth_region = frame[y_min:y_max, x_min:x_max].copy()
            
            # Validate quality
            is_valid, quality_score, issues = self.validate_viseme_quality(mouth_region)
            
            # Save viseme image
            viseme_path = self.output_dir / f"{phoneme}.png"
            cv2.imwrite(str(viseme_path), mouth_region)
            
            # Save verification image (full frame with ROI box)
            if self.config['extraction_params']['save_verification_images']:
                verification = frame.copy()
                cv2.rectangle(verification, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(verification, phoneme, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(verification, f"{timestamp:.2f}s | Q:{quality_score:.0f}", 
                           (x_min, y_max + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                verification_path = self.output_dir / f"{phoneme}_verification.png"
                cv2.imwrite(str(verification_path), verification)
            
            # Create VisemeData
            viseme_data = VisemeData(
                phoneme=phoneme,
                image=mouth_region,
                bounds=(x_min, y_min, x_max, y_max),
                timestamp=timestamp,
                frame_number=frame_num,
                quality_score=quality_score,
                is_valid=is_valid,
                issues=issues
            )
            
            # Print result
            status = "✓" if is_valid else "⚠"
            issue_str = f" ({', '.join(issues)})" if issues else ""
            print(f"  {status} {phoneme}: {mouth_region.shape} @ {timestamp:.2f}s | Q:{quality_score:.0f}{issue_str}")
            
            return viseme_data
            
        except Exception as e:
            print(f"  ✗ {phoneme}: Error - {e}")
            return None
    
    def extract_all_visemes(self) -> Dict[str, VisemeData]:
        """
        Extract all visemes from video
        
        Returns:
            Dict mapping phoneme -> VisemeData
        """
        viseme_library = {}
        timestamps = self.config['viseme_timestamps']
        
        print("Extracting visemes...\n")
        
        for phoneme in self.VISEME_PHONEMES:
            timestamp = timestamps.get(phoneme)
            
            if timestamp is None:
                print(f"  ⊘ {phoneme}: No timestamp configured (skipped)")
                continue
            
            viseme_data = self.extract_viseme(phoneme, timestamp)
            
            if viseme_data:
                viseme_library[phoneme] = viseme_data
        
        print(f"\n{'='*70}")
        print(f"Extraction complete: {len(viseme_library)}/{len(self.VISEME_PHONEMES)} visemes")
        print(f"{'='*70}\n")
        
        return viseme_library
    
    def generate_contact_sheet(self, viseme_library: Dict[str, VisemeData]):
        """Create visual contact sheet of all visemes"""
        if not viseme_library:
            print("No visemes to generate contact sheet")
            return
        
        cols = 5
        rows = (len(viseme_library) + cols - 1) // cols
        
        # Find max dimensions
        max_h = max(v.image.shape[0] for v in viseme_library.values())
        max_w = max(v.image.shape[1] for v in viseme_library.values())
        
        # Add padding for labels
        cell_h = max_h + 60
        cell_w = max_w + 30
        
        # Create canvas
        canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
        
        # Place visemes
        for idx, (phoneme, data) in enumerate(sorted(viseme_library.items())):
            row = idx // cols
            col = idx % cols
            
            y_offset = row * cell_h
            x_offset = col * cell_w
            
            # Center image in cell
            img_h, img_w = data.image.shape[:2]
            y_start = y_offset + (cell_h - img_h - 50) // 2
            x_start = x_offset + (cell_w - img_w) // 2
            
            # Place image
            canvas[y_start:y_start+img_h, x_start:x_start+img_w] = data.image
            
            # Add label
            label = f"{phoneme}"
            quality_label = f"Q:{data.quality_score:.0f}"
            
            label_y = y_start + img_h + 30
            label_x = x_offset + cell_w // 2 - 25
            
            cv2.putText(canvas, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(canvas, quality_label, (label_x, label_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Save contact sheet
        sheet_path = self.output_dir / "contact_sheet.png"
        cv2.imwrite(str(sheet_path), canvas)
        print(f"💾 Contact sheet saved: {sheet_path}")
    
    def save_library(self, viseme_library: Dict[str, VisemeData]):
        """Save viseme library and metadata"""
        # Save pickle (for runtime use)
        pickle_data = {
            phoneme: {
                'phoneme': data.phoneme,
                'image': data.image,
                'bounds': data.bounds,
                'timestamp': data.timestamp,
                'frame_number': data.frame_number,
                'quality_score': data.quality_score
            }
            for phoneme, data in viseme_library.items()
        }
        
        pickle_path = self.output_dir / "viseme_library.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        print(f"💾 Viseme library saved: {pickle_path}")
        
        # Save metadata (for review)
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'video_source': self.video_path,
            'video_properties': {
                'resolution': f"{self.width}x{self.height}",
                'fps': self.fps,
                'duration_seconds': self.total_frames / self.fps
            },
            'visemes': {
                phoneme: {
                    'timestamp': data.timestamp,
                    'frame_number': data.frame_number,
                    'roi_bounds': data.bounds,
                    'image_size': list(data.image.shape),
                    'quality_score': data.quality_score,
                    'is_valid': data.is_valid,
                    'issues': data.issues
                }
                for phoneme, data in viseme_library.items()
            },
            'summary': {
                'total_visemes': len(self.VISEME_PHONEMES),
                'successful': sum(1 for v in viseme_library.values() if v.is_valid),
                'warnings': sum(1 for v in viseme_library.values() if not v.is_valid),
                'failed': len(self.VISEME_PHONEMES) - len(viseme_library),
                'average_quality': np.mean([v.quality_score for v in viseme_library.values()])
            }
        }
        
        metadata_path = self.metadata_dir / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📝 Metadata saved: {metadata_path}")
    
    def generate_report(self, viseme_library: Dict[str, VisemeData]):
        """Generate extraction report"""
        total = len(self.VISEME_PHONEMES)
        successful = sum(1 for v in viseme_library.values() if v.is_valid)
        warnings = sum(1 for v in viseme_library.values() if not v.is_valid)
        failed = total - len(viseme_library)
        
        avg_quality = np.mean([v.quality_score for v in viseme_library.values()]) if viseme_library else 0
        min_quality = min([v.quality_score for v in viseme_library.values()]) if viseme_library else 0
        max_quality = max([v.quality_score for v in viseme_library.values()]) if viseme_library else 0
        
        report = f"""
{'='*70}
VISEME EXTRACTION REPORT
{'='*70}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video: {self.video_path}
Duration: {self.total_frames/self.fps:.2f}s | FPS: {self.fps} | Resolution: {self.width}x{self.height}

{'='*70}
EXTRACTION RESULTS
{'='*70}
Total visemes attempted: {total}
Successful: {successful} ({successful/total*100:.1f}%)
Warnings: {warnings} ({warnings/total*100:.1f}%)
Failed: {failed} ({failed/total*100:.1f}%)

{'='*70}
QUALITY METRICS
{'='*70}
Average quality score: {avg_quality:.1f}
Minimum quality: {min_quality:.1f}
Maximum quality: {max_quality:.1f}

{'='*70}
ISSUES
{'='*70}
"""
        
        # Failed extractions
        failed_phonemes = [p for p in self.VISEME_PHONEMES if p not in viseme_library]
        if failed_phonemes:
            report += "\nFAILED EXTRACTIONS:\n"
            for p in failed_phonemes:
                ts = self.config['viseme_timestamps'].get(p)
                report += f"  ✗ {p}: Failed at {ts}s\n"
        
        # Warnings
        warning_phonemes = [p for p, v in viseme_library.items() if not v.is_valid]
        if warning_phonemes:
            report += "\nWARNINGS:\n"
            for p in warning_phonemes:
                v = viseme_library[p]
                report += f"  ⚠ {p}: Q={v.quality_score:.1f} - {', '.join(v.issues)}\n"
        
        report += f"\n{'='*70}\n"
        report += "NEXT STEPS:\n"
        report += "1. Review contact sheet: output/visemes/contact_sheet.png\n"
        report += "2. Fix failed extractions (adjust timestamps)\n"
        report += "3. Re-run for low quality visemes if needed\n"
        report += "4. Proceed to RICo blending when satisfied\n"
        report += f"{'='*70}\n"
        
        print(report)
        
        # Save report
        report_path = self.metadata_dir / "extraction_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved: {report_path}")
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        self.face_mesh.close()


def main():
    """Main execution"""
    try:
        # Initialize extractor
        extractor = VisemeExtractor()
        
        # Extract visemes
        viseme_library = extractor.extract_all_visemes()
        
        # Generate outputs
        if viseme_library:
            extractor.save_library(viseme_library)
            
            if extractor.config['extraction_params']['save_contact_sheet']:
                extractor.generate_contact_sheet(viseme_library)
            
            extractor.generate_report(viseme_library)
        else:
            print("\n❌ No visemes extracted. Check timestamps in config/viseme_config.json")
        
        # Cleanup
        extractor.cleanup()
        
        print("\n✅ Extraction complete!")
        print("\nNext: Review contact sheet and proceed to RICo blending")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Placed video at: input/base_video_static.mp4")
        print("  2. Created config: config/viseme_config.json")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
