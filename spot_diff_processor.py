"""
spot_diff_processor_production.py
Split Spot the difference images and prepare them for annotation.
"""
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2

class SpotDiffProcessorProduction:
    """
    Production processor for 300K images on Google Colab.
    Optimized for speed, memory, and reliability.
    """
    
    def __init__(self, 
                 source_dir: str, 
                 output_base: str = "processed_pairs",
                 checkpoint_file: str = "checkpoint.json",
                 batch_size: int = 100):
        
        self.source_dir = Path(source_dir)
        self.output_base = Path(output_base)
        self.checkpoint_file = Path(checkpoint_file)
        self.batch_size = batch_size
        
        # Create output directory
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint state
        self.checkpoint_state = {
            "processed_files": {},  # lowercase_name -> pair_id
            "failed_files": {},
            "stats": {
                "total_files": 0,
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "vertical_splits": 0,
                "horizontal_splits": 0,
                "start_time": datetime.now().isoformat()
            }
        }
        
        # Load existing checkpoint
        self._load_checkpoint()
        
        # Get UNIQUE image files (case-insensitive)
        self.image_files = self._get_unique_image_files()
        
        print(f"Found {len(self.image_files)} unique images")
        print(f"Already processed: {len(self.checkpoint_state['processed_files'])}")
    
    def _get_unique_image_files(self) -> List[Path]:
        """Get unique image files (case-insensitive)."""
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        seen = set()
        unique_files = []
        
        for ext in extensions:
            for file_path in self.source_dir.glob(f"*{ext}"):
                # Case-insensitive deduplication
                lower_name = file_path.name.lower()
                if lower_name not in seen:
                    seen.add(lower_name)
                    unique_files.append(file_path)
            
            # Also check uppercase extensions
            for file_path in self.source_dir.glob(f"*{ext.upper()}"):
                lower_name = file_path.name.lower()
                if lower_name not in seen:
                    seen.add(lower_name)
                    unique_files.append(file_path)
        
        # Sort for consistent ordering
        return sorted(unique_files, key=lambda x: x.name.lower())
    
    def _load_checkpoint(self):
        """Load checkpoint state."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint_state = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint state."""
        self.checkpoint_state["stats"]["last_save"] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
    
    def _safe_crop_borders(self, image: np.ndarray) -> np.ndarray:
        """
        SAFE cropping: Only remove pure black/white borders.
        Returns image unchanged if border has content.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Check each border for uniformity
        borders = {'top': 0, 'bottom': h, 'left': 0, 'right': w}
        
        # Check top border (max 10% of height)
        max_border = min(100, h // 10)
        for y in range(max_border):
            row = gray[y, :]
            if np.std(row) > 20:  # Border has content, stop
                borders['top'] = y
                break
            if y == max_border - 1:
                borders['top'] = max_border
        
        # Check bottom border
        for y in range(h-1, max(h-100, h - h//10), -1):
            row = gray[y, :]
            if np.std(row) > 20:
                borders['bottom'] = y + 1
                break
        
        # Check left border
        for x in range(min(100, w // 10)):
            col = gray[:, x]
            if np.std(col) > 20:
                borders['left'] = x
                break
        
        # Check right border
        for x in range(w-1, max(w-100, w - w//10), -1):
            col = gray[:, x]
            if np.std(col) > 20:
                borders['right'] = x + 1
                break
        
        # Only crop if we found significant borders
        if (borders['top'] > 10 or borders['bottom'] < h - 10 or 
            borders['left'] > 10 or borders['right'] < w - 10):
            
            cropped = image[borders['top']:borders['bottom'], 
                           borders['left']:borders['right']]
            
            if cropped.size > 0:
                # Ensure we didn't crop too much
                h_new, w_new = cropped.shape[:2]
                if h_new > h * 0.8 and w_new > w * 0.8:
                    return cropped
        
        return image
    
    def _find_dividing_line_simple(self, gray: np.ndarray) -> Tuple[str, int]:
        """
        SIMPLE and ROBUST divider detection.
        Focuses on edge density in center region.
        """
        h, w = gray.shape
        
        # Enhance edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Define center region (40-60%)
        center_x_start = int(w * 0.4)
        center_x_end = int(w * 0.6)
        center_y_start = int(h * 0.4)
        center_y_end = int(h * 0.6)
        
        # Calculate vertical edge density in center region
        vertical_edges = edges[:, center_x_start:center_x_end]
        vertical_density = np.sum(vertical_edges, axis=0) / h
        
        # Calculate horizontal edge density in center region
        horizontal_edges = edges[center_y_start:center_y_end, :]
        horizontal_density = np.sum(horizontal_edges, axis=1) / w
        
        # Find peaks
        if len(vertical_density) > 0:
            v_peak_idx = np.argmax(vertical_density)
            v_peak_value = vertical_density[v_peak_idx]
            v_position = center_x_start + v_peak_idx
        else:
            v_peak_value = 0
            v_position = w // 2
        
        if len(horizontal_density) > 0:
            h_peak_idx = np.argmax(horizontal_density)
            h_peak_value = horizontal_density[h_peak_idx]
            h_position = center_y_start + h_peak_idx
        else:
            h_peak_value = 0
            h_position = h // 2
        
        # Decision: Choose clearer divider
        if v_peak_value > 0.05 and v_peak_value > h_peak_value * 1.2:
            return "vertical", v_position
        elif h_peak_value > 0.05 and h_peak_value > v_peak_value * 1.2:
            return "horizontal", h_position
        else:
            # Default based on aspect ratio
            return ("vertical", w // 2) if w > h else ("horizontal", h // 2)
    
    def _split_with_margin(self, image: np.ndarray, orientation: str, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split with 5-pixel margin."""
        margin = 5
        
        if orientation == 'vertical':
            pos = max(margin, min(image.shape[1] - margin, pos))
            img1 = image[:, :pos - margin]
            img2 = image[:, pos + margin:]
        else:
            pos = max(margin, min(image.shape[0] - margin, pos))
            img1 = image[:pos - margin, :]
            img2 = image[pos + margin:, :]
        
        return img1, img2
    
    def _find_top_differences(self, img1: np.ndarray, img2: np.ndarray, top_n: int = 5) -> List[Dict[str, int]]:
        """Find top N differences with area filtering."""
        # Resize to common size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h, w = min(h1, h2), min(w1, w2)
        
        if h == 0 or w == 0:
            return []
        
        img1_resized = cv2.resize(img1, (w, h))
        img2_resized = cv2.resize(img2, (w, h))
        
        # Convert to grayscale
        if len(img1_resized.shape) == 3:
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1_resized
            gray2 = img2_resized
        
        # Absolute difference with threshold
        diff = cv2.absdiff(gray1, gray2)
        _, diff_binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        diff_clean = cv2.morphologyEx(diff_binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 <= area <= (h * w * 0.1):  # Reasonable size limits
                x, y, w_box, h_box = cv2.boundingRect(contour)
                boxes.append({
                    'x': int(x), 'y': int(y), 
                    'w': int(w_box), 'h': int(h_box),
                    'area': int(area)
                })
        
        # Sort by area and take top N
        boxes.sort(key=lambda b: b['area'], reverse=True)
        top_boxes = boxes[:top_n]
        
        # Remove area field from output
        for box in top_boxes:
            if 'area' in box:
                del box['area']
        
        return top_boxes
    
    def process_single_image(self, image_path: Path, idx: int) -> Optional[Dict[str, Any]]:
        """Process a single image with error handling."""
        try:
            # Check if already processed (case-insensitive)
            lower_name = image_path.name.lower()
            if lower_name in self.checkpoint_state["processed_files"]:
                pair_id = self.checkpoint_state["processed_files"][lower_name]
                print(f"Skipping {image_path.name} - already processed as {pair_id}")
                self.checkpoint_state["stats"]["skipped"] += 1
                return None
            
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to read image")
            
            # Safe cropping
            cleaned = self._safe_crop_borders(img)
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            
            # Find dividing line
            orientation, split_pos = self._find_dividing_line_simple(gray)
            
            # Split image
            img1, img2 = self._split_with_margin(cleaned, orientation, split_pos)
            
            # Find differences
            differences = self._find_top_differences(img1, img2, top_n=5)
            
            # Generate pair ID
            pair_id = f"{idx:07d}"
            output_dir = self.output_base / pair_id
            output_dir.mkdir(exist_ok=True)
            
            # Save images
            cv2.imwrite(str(output_dir / "1.png"), img1)
            cv2.imwrite(str(output_dir / "2.png"), img2)
            
            # Prepare metadata
            meta = {
                "id": pair_id,
                "source_file": image_path.name,
                "split_orientation": orientation,
                "split_position": int(split_pos),
                "differences": differences,
                "processing_time": datetime.now().isoformat()
            }
            
            # Save metadata
            with open(output_dir / "meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Update stats
            if orientation == "vertical":
                self.checkpoint_state["stats"]["vertical_splits"] += 1
            else:
                self.checkpoint_state["stats"]["horizontal_splits"] += 1
            
            # Mark as processed
            self.checkpoint_state["processed_files"][lower_name] = pair_id
            
            print(f"✓ {pair_id}: {orientation} split, {len(differences)} diffs")
            
            return meta
            
        except Exception as e:
            error_msg = f"Failed {image_path.name}: {str(e)[:50]}"
            print(f"✗ {error_msg}")
            
            self.checkpoint_state["failed_files"][image_path.name] = {
                "error": str(e),
                "time": datetime.now().isoformat()
            }
            
            return {"error": str(e)}
    
    def run(self, limit: Optional[int] = None):
        """Main processing loop optimized for Colab."""
        print("=" * 60)
        print("PRODUCTION PROCESSOR - Google Colab Ready")
        print("=" * 60)
        
        total = len(self.image_files)
        processed = 0
        failed = 0
        
        for idx, image_path in enumerate(self.image_files):
            # Apply limit for testing
            if limit and (processed + failed) >= limit:
                break
            
            # Process image
            result = self.process_single_image(image_path, idx)
            
            if result:
                if "error" in result:
                    failed += 1
                    self.checkpoint_state["stats"]["failed"] += 1
                else:
                    processed += 1
                    self.checkpoint_state["stats"]["processed"] += 1
            else:
                # Skipped
                continue
            
            # Save checkpoint every batch
            if (processed + failed) % self.batch_size == 0:
                self.checkpoint_state["stats"]["total_files"] = total
                self._save_checkpoint()
                print(f"\nCheckpoint saved. Progress: {processed+failed}/{total}")
        
        # Final save
        self.checkpoint_state["stats"]["total_files"] = total
        self.checkpoint_state["stats"]["end_time"] = datetime.now().isoformat()
        self._save_checkpoint()
        
        # Final report
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        stats = self.checkpoint_state["stats"]
        print(f"Total files: {total}")
        print(f"Successfully processed: {processed}")
        print(f"Failed: {failed}")
        print(f"Skipped (already done): {stats.get('skipped', 0)}")
        print(f"Vertical splits: {stats.get('vertical_splits', 0)}")
        print(f"Horizontal splits: {stats.get('horizontal_splits', 0)}")
        print(f"\nOutput ready for Script 2 (VLM Annotator)")

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production Spot-the-Difference Processor for Google Colab',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--source', '-s', required=True,
                       help='Source directory with images')
    parser.add_argument('--output', '-o', default='processed_pairs',
                       help='Output directory')
    parser.add_argument('--checkpoint', '-c', default='checkpoint.json',
                       help='Checkpoint file')
    parser.add_argument('--limit', '-l', type=int,
                       help='Limit number of images (for testing)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    processor = SpotDiffProcessorProduction(
        source_dir=args.source,
        output_base=args.output,
        checkpoint_file=args.checkpoint,
        batch_size=args.batch_size
    )
    
    processor.run(limit=args.limit)

if __name__ == "__main__":
    main()