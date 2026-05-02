#!/usr/bin/env python3
# =============================================================================
# test_single.py — Test a single image and show detailed results
#
# Usage:
#   python test_single.py <image_path> [--show-ground-truth]
# =============================================================================

import argparse
import csv
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline


def load_ground_truth(annotations_file: str) -> dict:
    """Load ground truth annotations."""
    gt = {}
    try:
        with open(annotations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename'].strip()
                if filename not in gt:
                    gt[filename] = {
                        'class': row['class'].replace('-block', ''),
                        'xmin': int(row['xmin']),
                        'ymin': int(row['ymin']),
                        'xmax': int(row['xmax']),
                        'ymax': int(row['ymax']),
                    }
    except FileNotFoundError:
        print(f"Warning: Could not find annotations file: {annotations_file}")
    return gt


def main():
    parser = argparse.ArgumentParser(description="Test a single image")
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument(
        '--show-ground-truth',
        action='store_true',
        help='Show ground truth if available'
    )
    parser.add_argument(
        '--annotations',
        default='data/train/_annotations.csv',
        help='Path to annotations CSV'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Save debug images'
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Load ground truth if requested
    ground_truth = None
    if args.show_ground_truth:
        gt_dict = load_ground_truth(args.annotations)
        ground_truth = gt_dict.get(image_path.name)
    
    # Run detection
    print(f"\n{'='*70}")
    print(f"  TESTING: {image_path.name}")
    print(f"{'='*70}\n")
    
    try:
        result = run_pipeline(str(image_path), debug=args.debug)
        
        print(f"\n{'='*70}")
        print(f"  DETECTION RESULTS")
        print(f"{'='*70}")
        print(f"  Active Piece Type:  {result.get('piece_type', 'UNKNOWN')}")
        print(f"  Piece Position:     row={result.get('piece_pos', [None, None])[0]}, "
              f"col={result.get('piece_pos', [None, None])[1]}")
        print(f"  Next Piece Type:    {result.get('next_type', 'UNKNOWN')}")
        
        if ground_truth:
            detected = result.get('piece_type', 'UNKNOWN')
            expected = ground_truth['class']
            match = detected == expected
            
            print(f"\n{'─'*70}")
            print(f"  GROUND TRUTH COMPARISON")
            print(f"{'─'*70}")
            print(f"  Expected Piece:     {expected}")
            print(f"  Detected Piece:     {detected}")
            print(f"  Ground Truth BBox:  ({ground_truth['xmin']}, {ground_truth['ymin']}) "
                  f"to ({ground_truth['xmax']}, {ground_truth['ymax']})")
            print(f"  Result:             {'✓ CORRECT' if match else '✗ MISMATCH'}")
        
        print(f"\n{'='*70}\n")
    
    except Exception as e:
        print(f"Error during detection: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
