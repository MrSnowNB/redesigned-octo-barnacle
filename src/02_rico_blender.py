"""
02_rico_blender.py

RICo Blender - Seamless Viseme Layer Compositing
Blend extracted viseme images onto base video with imperceptible seams

Usage:
    python src/02_rico_blender.py --text "Hello, how are you?" --output test_output.mp4

Requirements:
    - Viseme library: output/visemes/viseme_library.pkl
    - Config: config/blending_config.json
    - Phoneme map: config/phoneme_map.json
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import sys



class RICOBlender:
    """Seamless viseme layer compositing engine"""

    def __init__(self, config_path: str = 'config/blending_config.json'):
        """Initialize blender with configuration"""
        with open(config_path) as f:
            self.config = json.load(f)

        # Load phoneme map
        with open('config/phoneme_map.json') as f:
            self.phoneme_map = json.load(f)

        # Initialize MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("RICo Blender initialized")

    def load_viseme_library(self, library_path: str) -> Dict:
        """Load extracted viseme library"""
        with open(library_path, 'rb') as f:
            return pickle.load(f)

    def alpha_feather_blend(self, base_frame: np.ndarray, viseme_img: np.ndarray,
                          roi_bounds: Tuple[int, int, int, int],
                          feather_percent: float = 0.25) -> np.ndarray:
        """Fast alpha blending with feathered edges"""
        x_min, y_min, x_max, y_max = roi_bounds
        h, w = y_max - y_min, x_max - x_min

        # Resize viseme to ROI size
        viseme_resized = cv2.resize(viseme_img, (w, h))

        # Create feathered alpha mask
        mask = self._create_feather_mask(w, h, feather_percent)

        # Extract base ROI
        base_roi = base_frame[y_min:y_max, x_min:x_max]

        # Alpha blend
        blended_roi = (base_roi * (1 - mask) + viseme_resized * mask).astype(np.uint8)

        # Copy back to frame
        result = base_frame.copy()
        result[y_min:y_max, x_min:x_max] = blended_roi

        return result

    def _create_feather_mask(self, width: int, height: int, feather_percent: float) -> np.ndarray:
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

    def poisson_blend(self, base_frame: np.ndarray, viseme_img: np.ndarray,
                     roi_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Seamless Poisson blending using OpenCV's seamlessClone"""
        x_min, y_min, x_max, y_max = roi_bounds
        h, w = y_max - y_min, x_max - x_min

        # Resize viseme
        viseme_resized = cv2.resize(viseme_img, (w, h))

        # Create mask (white ellipse for mouth region)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 2 - 10, h // 2 - 10)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # Feather mask edges for smoother blending
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

    def detect_mouth_params(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect mouth position, rotation, and scale from frame

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

        padding = 25
        x_min = max(0, min(xs) - padding)
        y_min = max(0, min(ys) - padding)
        x_max = min(w, max(xs) + padding)
        y_max = min(h, max(ys) + padding)

        # Calculate center
        center_x = (min(xs) + max(xs)) // 2
        center_y = (min(ys) + max(ys)) // 2

        # Calculate angle and scale
        left_corner = mouth_points[0]  # 61
        right_corner = mouth_points[1]  # 291
        dx = right_corner[0] - left_corner[0]
        dy = right_corner[1] - left_corner[1]
        angle = np.degrees(np.arctan2(dy, dx))
        scale = np.sqrt(dx**2 + dy**2)

        return {
            'center': (center_x, center_y),
            'angle': angle,
            'scale': scale,
            'bounds': (x_min, y_min, x_max, y_max)
        }

    def map_phoneme_to_viseme(self, phoneme: str) -> Optional[str]:
        """
        Map TTS phoneme to viseme ID

        Args:
            phoneme: ARPAbet phoneme (e.g., 'AH0', 'T')

        Returns:
            Viseme ID or None for silence
        """
        # Remove stress markers
        phoneme_base = ''.join(c for c in phoneme if c.isalpha())

        # Check silence
        if phoneme in self.phoneme_map['silence_phonemes']:
            return None

        # Map to viseme
        return self.phoneme_map['phoneme_to_viseme_map'].get(phoneme_base, 'AH')

    def generate_test_video(self, text: str, output_path: str, library_path: str = 'output/visemes/viseme_library.pkl'):
        """Generate test video with simple phoneme-to-viseme mapping"""
        print(f"Generating test video for: '{text}'")
        print(f"Output: {output_path}")

        # Load viseme library
        try:
            viseme_library = self.load_viseme_library(library_path)
            print(f"Loaded {len(viseme_library)} visemes")
        except FileNotFoundError:
            print(f"Error: Viseme library not found at {library_path}")
            print("Run viseme extraction first: python run_pipeline.py extract")
            return

        # Load base video
        base_video_path = 'input/base_video_static.mp4'
        if not Path(base_video_path).exists():
            print(f"Error: Base video not found at {base_video_path}")
            return

        cap = cv2.VideoCapture(base_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open base video {base_video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Base video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup output video
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Simple phoneme simulation: cycle through visemes based on text
        viseme_keys = list(viseme_library.keys())
        text_length = len(text)
        frames_per_viseme = max(1, total_frames // text_length)

        print(f"Generating {total_frames} frames with {len(viseme_keys)} visemes")

        frame_count = 0
        viseme_index = 0

        while frame_count < total_frames:
            ret, base_frame = cap.read()
            if not ret:
                # Loop back to beginning if needed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, base_frame = cap.read()
                if not ret:
                    break

            # Select viseme (simple cycling based on text position)
            current_viseme_key = viseme_keys[viseme_index % len(viseme_keys)]

            # Detect mouth in current frame
            mouth_params = self.detect_mouth_params(base_frame)

            if mouth_params and current_viseme_key in viseme_library:
                viseme_data = viseme_library[current_viseme_key]
                viseme_img = viseme_data['image']

                # Choose blending mode
                blend_mode = self.config.get('blending_mode', 'alpha')

                if blend_mode == 'poisson':
                    blended_frame = self.poisson_blend(base_frame, viseme_img, mouth_params['bounds'])
                else:
                    feather_percent = self.config.get('feather_percent', 0.25)
                    blended_frame = self.alpha_feather_blend(base_frame, viseme_img, mouth_params['bounds'], feather_percent)

                out.write(blended_frame)
            else:
                # No face detected or viseme missing, use base frame
                out.write(base_frame)

            frame_count += 1

            # Advance viseme based on text length
            if frame_count % frames_per_viseme == 0:
                viseme_index += 1

        cap.release()
        out.release()

        print(f"✅ Test video generated: {output_path}")
        print(f"   Duration: {total_frames/fps:.2f}s")
        print(f"   Visemes used: {len(viseme_keys)}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='RICo Viseme Blender')
    parser.add_argument('--text', type=str, default="Hello, world!",
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output/test_videos/test_output.mp4',
                       help='Output video path')
    parser.add_argument('--library', type=str, default='output/visemes/viseme_library.pkl',
                       help='Viseme library path')

    args = parser.parse_args()

    try:
        blender = RICOBlender()
        blender.generate_test_video(args.text, args.output, args.library)
        print("Blending complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
