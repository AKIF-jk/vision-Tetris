#!/usr/bin/env python3
# =============================================================================
# test_validation.py — Validate detections against ground truth annotations
#
# Usage:
#   python test_validation.py                  # Run all tests
#   python test_validation.py --verbose        # Show detailed results
#   python test_validation.py --save-results   # Save detailed results to CSV
# =============================================================================

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_pipeline


class TestValidator:
    def __init__(self, train_dir: str, annotations_file: str, verbose: bool = False):
        self.train_dir = Path(train_dir)
        self.annotations_file = Path(annotations_file)
        self.verbose = verbose
        self.results = []
        self.ground_truth = {}
        
        # Load ground truth annotations
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Load ground truth from CSV into a dictionary."""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename'].strip()
                # Store the ground truth (class and bounding box)
                if filename not in self.ground_truth:
                    self.ground_truth[filename] = {
                        'class': row['class'],
                        'xmin': int(row['xmin']),
                        'ymin': int(row['ymin']),
                        'xmax': int(row['xmax']),
                        'ymax': int(row['ymax']),
                    }
        
        print(f"✓ Loaded {len(self.ground_truth)} ground truth annotations\n")
    
    def _normalize_class_name(self, class_name: str) -> str:
        """
        Normalize class name to match between ground truth and detection.
        Ground truth: "I-block", "O-block", etc.
        Detection: "I", "O", etc.
        """
        if class_name.endswith('-block'):
            return class_name.replace('-block', '')
        return class_name
    
    def _get_image_files(self):
        """Get all image files from train directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in self.train_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        return sorted(images)
    
    def run_tests(self):
        """Run validation tests on all training images."""
        image_files = self._get_image_files()
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"No images found in {self.train_dir}")
            return
        
        print(f"{'='*70}")
        print(f"  VALIDATION TEST — {total_images} images")
        print(f"{'='*70}\n")
        
        correct = 0
        mismatches = []
        
        for idx, image_path in enumerate(image_files, 1):
            filename = image_path.name
            
            # Check if ground truth exists
            if filename not in self.ground_truth:
                if self.verbose:
                    print(f"[{idx:3d}/{total_images}] ⚠ {filename:50s} — NO GROUND TRUTH")
                continue
            
            gt_class = self.ground_truth[filename]['class']
            gt_class_normalized = self._normalize_class_name(gt_class)
            
            try:
                # Run detection pipeline (suppress debug output)
                result = run_pipeline(str(image_path), debug=False)
                detected_class = result.get('piece_type', 'UNKNOWN')
                detected_class_normalized = self._normalize_class_name(detected_class)
                
                # Check if match
                is_match = detected_class_normalized == gt_class_normalized
                if is_match:
                    correct += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ MISMATCH"
                    mismatches.append({
                        'filename': filename,
                        'ground_truth': gt_class_normalized,
                        'detected': detected_class_normalized,
                    })
                
                if self.verbose:
                    print(f"[{idx:3d}/{total_images}] {status:12s} {filename:50s} "
                          f"GT: {gt_class_normalized:2s}  →  Detected: {detected_class_normalized:2s}")
                else:
                    # Progress indicator
                    if idx % 10 == 0:
                        print(f"  Progress: {idx}/{total_images} ({100*idx/total_images:.0f}%)")
                
                # Store result
                self.results.append({
                    'filename': filename,
                    'ground_truth_class': gt_class_normalized,
                    'detected_class': detected_class_normalized,
                    'match': is_match,
                    'ground_truth_bbox': (
                        self.ground_truth[filename]['xmin'],
                        self.ground_truth[filename]['ymin'],
                        self.ground_truth[filename]['xmax'],
                        self.ground_truth[filename]['ymax'],
                    ),
                })
            
            except Exception as e:
                print(f"[{idx:3d}/{total_images}] ✗ ERROR     {filename:50s} — {str(e)}")
        
        # Print summary
        self._print_summary(correct, total_images, mismatches)
    
    def _print_summary(self, correct: int, total: int, mismatches: list):
        """Print test summary with statistics."""
        accuracy = 100 * correct / total if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"  TEST SUMMARY")
        print(f"{'='*70}")
        print(f"  Total images tested: {total}")
        print(f"  Correct detections:  {correct}/{total}")
        print(f"  Accuracy:            {accuracy:.1f}%")
        print(f"\n")
        
        # Per-class statistics
        if self.results:
            self._print_per_class_stats()
        
        # Show mismatches
        if mismatches:
            print(f"  MISMATCHES ({len(mismatches)}):")
            print(f"  {'-'*70}")
            for mismatch in mismatches[:20]:  # Show first 20
                print(f"    {mismatch['filename']:50s} "
                      f"Expected: {mismatch['ground_truth']:2s}  "
                      f"Got: {mismatch['detected']:2s}")
            if len(mismatches) > 20:
                print(f"    ... and {len(mismatches) - 20} more mismatches")
        
        print(f"\n{'='*70}\n")
    
    def _print_per_class_stats(self):
        """Print per-class accuracy statistics."""
        stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        confusion = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            gt_class = result['ground_truth_class']
            detected_class = result['detected_class']
            
            stats[gt_class]['total'] += 1
            confusion[gt_class][detected_class] += 1
            
            if result['match']:
                stats[gt_class]['correct'] += 1
        
        print(f"  PER-CLASS STATISTICS:")
        print(f"  {'-'*70}")
        print(f"  {'Class':<10} {'Correct':<12} {'Total':<10} {'Accuracy':<12}")
        print(f"  {'-'*70}")
        
        for class_name in sorted(stats.keys()):
            stat = stats[class_name]
            acc = 100 * stat['correct'] / stat['total'] if stat['total'] > 0 else 0
            print(f"  {class_name:<10} {stat['correct']:<12} {stat['total']:<10} {acc:>6.1f}%")
        
        print(f"  {'-'*70}\n")
        
        # Confusion matrix
        if len(stats) <= 7:  # Only show if reasonable size
            self._print_confusion_matrix(confusion, stats)
    
    def _print_confusion_matrix(self, confusion: dict, stats: dict):
        """Print confusion matrix."""
        classes = sorted(stats.keys())
        print(f"  CONFUSION MATRIX:")
        print(f"  {'-'*70}")
        print(f"  {'':>5} " + "".join(f"{c:>6}" for c in classes))
        print(f"  {'-'*70}")
        for gt_class in classes:
            row = f"  {gt_class:>5} "
            for detected_class in classes:
                count = confusion[gt_class][detected_class]
                row += f"{count:>6}"
            print(row)
        print(f"  {'-'*70}\n")
    
    def save_results_to_csv(self, output_file: str = 'validation_results.csv'):
        """Save detailed results to CSV."""
        if not self.results:
            print("No results to save.")
            return
        
        output_path = Path(output_file)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=[
                    'filename',
                    'ground_truth_class',
                    'detected_class',
                    'match',
                    'xmin', 'ymin', 'xmax', 'ymax'
                ]
            )
            writer.writeheader()
            
            for result in self.results:
                bbox = result['ground_truth_bbox']
                writer.writerow({
                    'filename': result['filename'],
                    'ground_truth_class': result['ground_truth_class'],
                    'detected_class': result['detected_class'],
                    'match': result['match'],
                    'xmin': bbox[0],
                    'ymin': bbox[1],
                    'xmax': bbox[2],
                    'ymax': bbox[3],
                })
        
        print(f"✓ Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Tetris piece detection against ground truth"
    )
    parser.add_argument(
        '--train-dir',
        default='data/train',
        help='Path to training directory (default: data/train)'
    )
    parser.add_argument(
        '--annotations',
        default='data/train/_annotations.csv',
        help='Path to annotations CSV (default: data/train/_annotations.csv)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed results for each image'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to CSV'
    )
    parser.add_argument(
        '--output',
        default='validation_results.csv',
        help='Output file for results (default: validation_results.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        validator = TestValidator(
            args.train_dir,
            args.annotations,
            verbose=args.verbose
        )
        validator.run_tests()
        
        if args.save_results:
            validator.save_results_to_csv(args.output)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
