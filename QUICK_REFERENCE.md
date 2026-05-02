# Testing Quick Reference

## One-Liners

### Get Current Performance

```bash
python3 test_validation.py --save-results && python3 analyze_results.py
```

### Test Single Image

```bash
python3 test_single.py data/train/img_501_jpg.rf.8b68739b5f01a3a3b036f8975a50172e.jpg --show-ground-truth
```

### Verbose Testing (see each result)

```bash
python3 test_validation.py --verbose
```

### Export Analysis Report

```bash
python3 analyze_results.py --export report.txt
```

## Common Tasks

### 1. Check overall accuracy

```bash
python3 test_validation.py
```

→ Shows summary statistics at the end

### 2. Find worst performing piece type

```bash
python3 analyze_results.py
```

→ Look for "Worst performing class"

### 3. Debug why a specific image failed

```bash
python3 test_single.py data/train/img_500_jpg.rf.120f62879407bfa8889e1e72a88c6043.jpg --show-ground-truth
```

→ Shows ground truth vs detected with ✓ or ✗

### 4. See which pieces are confused with each other

```bash
python3 analyze_results.py | grep -A 10 "FREQUENTLY CONFUSED"
```

→ Shows pair confusions and percentages

### 5. Track improvement over time

```bash
# Before making changes
python3 test_validation.py --save-results --output baseline.csv

# Make improvements...

# After changes
python3 test_validation.py --save-results --output improved.csv

# Compare
python3 analyze_results.py --input improved.csv
```

## File Locations

| File                     | Purpose                  |
| ------------------------ | ------------------------ |
| `test_validation.py`     | Full dataset test        |
| `test_single.py`         | Debug single image       |
| `analyze_results.py`     | Generate analysis report |
| `validation_results.csv` | Test results output      |
| `TESTING.md`             | Full documentation       |
| `TEST_SUMMARY.md`        | Overview of scripts      |

## Key Metrics

- **Overall Accuracy**: Percentage of all images correctly classified
- **Per-Class Accuracy**: How well each piece type is detected (I, O, T, S, Z, J, L)
- **Confusion Matrix**: Shows which pieces are confused with which
- **Common Confusions**: Which piece pairs are most often mixed up

## Current Baseline

```
✓ Runs: 108/108 images
✓ Correct: 36 detections
✓ Accuracy: 33.3%

Best:  O-blocks (75.0%)
Worst: T-blocks (0.0%)
```

## Tips

1. **Before running tests**: Make sure all images exist in `data/train/`
2. **Debug mode**: Use `test_single.py --debug` to see processing steps
3. **Batch results**: Use `--save-results` to save CSV for later analysis
4. **Compare versions**: Save results with different names to track improvements
5. **Identify patterns**: Use `analyze_results.py` to find systematic issues

---

**Start here**: `python3 test_validation.py --verbose` to see detailed results
