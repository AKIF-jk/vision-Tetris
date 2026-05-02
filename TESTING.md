# Testing & Validation Guide

This directory contains scripts to test and validate your Tetris piece detection program against ground truth annotations.

## Quick Start

### 1. Run Full Validation Test

```bash
python3 test_validation.py
```

This runs the detection pipeline on all training images and compares results against ground truth:

- Outputs overall accuracy
- Shows per-class statistics
- Generates confusion matrix
- Saves results to `validation_results.csv`

### 2. Analyze Results

```bash
python3 analyze_results.py
```

Generates detailed analysis report with:

- Overall performance metrics
- Per-class accuracy breakdown
- Most common misclassifications
- Confusion matrix
- Frequently confused class pairs
- Recommendations for improvement

### 3. Test Single Image

```bash
python3 test_single.py data/train/img_500_jpg.rf.120f62879407bfa8889e1e72a88c6043.jpg --show-ground-truth
```

Tests a single image and displays:

- Detected piece type
- Detected position/rotation
- Ground truth (if available)
- Whether detection matches ground truth

## Script Usage

### test_validation.py

**Options:**

- `--train-dir PATH`: Directory containing training images (default: `data/train`)
- `--annotations PATH`: Path to annotations CSV (default: `data/train/_annotations.csv`)
- `--verbose, -v`: Show detailed results for each image
- `--save-results`: Save results to CSV file
- `--output FILE`: Output CSV filename (default: `validation_results.csv`)

**Examples:**

```bash
# Run with verbose output
python3 test_validation.py --verbose

# Save results to custom file
python3 test_validation.py --save-results --output my_results.csv

# Test specific directory
python3 test_validation.py --train-dir data/valid
```

### analyze_results.py

**Options:**

- `--input, -i FILE`: Input CSV with results (default: `validation_results.csv`)
- `--export, -e FILE`: Export analysis report to text file

**Examples:**

```bash
# Analyze the default results file
python3 analyze_results.py

# Analyze custom results file
python3 analyze_results.py --input my_results.csv

# Export report to file
python3 analyze_results.py --export report.txt
```

### test_single.py

**Options:**

- `--show-ground-truth`: Display ground truth information
- `--annotations PATH`: Path to annotations CSV
- `--debug, -d`: Save debug images to output/debug/

**Examples:**

```bash
# Test with ground truth comparison
python3 test_single.py path/to/image.jpg --show-ground-truth

# Save debug images
python3 test_single.py path/to/image.jpg --debug
```

## Output Files

- **validation_results.csv**: Detailed per-image results with ground truth comparison
  - Columns: filename, ground_truth_class, detected_class, match, xmin, ymin, xmax, ymax

## Understanding Results

### Accuracy Metrics

- **Overall Accuracy**: Percentage of all images correctly classified
- **Per-Class Accuracy**: Accuracy for each Tetris piece type (I, O, T, S, Z, J, L)

### Confusion Matrix

- Rows represent ground truth class
- Columns represent detected class
- Diagonal values = correct detections
- Off-diagonal = misclassifications

### Most Common Misclassifications

Shows which piece types are most often confused with each other, helping identify:

- Pieces with similar shapes that your algorithm struggles to distinguish
- Areas for algorithm improvement

## Example Results Summary

```
Total Images:        108
Correct Detections:  36
Accuracy:            33.3%

PER-CLASS ACCURACY:
  I: 22.2%  (2/9)
  O: 75.0%  (9/12)
  T:  0.0%  (0/18)
  S: 37.5%  (6/16)
  Z: 56.5% (13/23)
  J: 13.3%  (2/15)
  L: 28.6%  (4/14)

MOST COMMON CONFUSIONS:
  I ↔ O: 16 times (22.5% of errors)
  O ↔ S: 14 times (19.7% of errors)
  O ↔ T: 14 times (19.7% of errors)
```

## Improving Detection Accuracy

Based on the analysis, focus on:

1. **Worst performing classes** (0% accuracy): Review shape detection for T-pieces
2. **Frequently confused pairs**: Improve feature extraction to better distinguish I↔O, O↔S, O↔T
3. **Overall accuracy**: Consider:
   - Adjusting piece shape matching thresholds
   - Improving color-based classification
   - Enhancing Hu moments computation
   - Tuning detection parameters in `config.py`

## Workflow

1. **Baseline**: Run `python3 test_validation.py --save-results` to get baseline accuracy
2. **Analyze**: Run `python3 analyze_results.py` to identify problem areas
3. **Debug**: Use `python3 test_single.py <image> --show-ground-truth --debug` to inspect specific failures
4. **Improve**: Adjust detection algorithm based on findings
5. **Validate**: Re-run full validation to measure improvement
6. **Compare**: Run `python3 analyze_results.py` to see before/after comparison

---

## Data Format

### Annotations CSV (\_annotations.csv)

```
filename,width,height,class,xmin,ymin,xmax,ymax
img_500.jpg,640,640,S-block,382,71,472,170
img_501.jpg,640,640,S-block,211,42,341,107
```

### Results CSV (validation_results.csv)

```
filename,ground_truth_class,detected_class,match,xmin,ymin,xmax,ymax
img_500.jpg,S,J,False,382,71,472,170
img_501.jpg,S,S,True,211,42,341,107
```
