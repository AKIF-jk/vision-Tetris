# Virtual Environment & Command Reference

## Virtual Environment Setup

### First Time Setup
```bash
# Navigate to project directory
cd /home/faran/Desktop/Semester06/DIP/Project/vision-Tetris

# Activate the virtual environment
source .venv/bin/activate

# Verify activation (your prompt should show (.venv) prefix)
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## Running the Main Pipeline

### Basic Command (Requires venv activation)
After activating the venv with `source .venv/bin/activate`:
```bash
python main.py --input data/train/sample_screenshot.jpg
```

### Full Path Command (No activation needed)
```bash
.venv/bin/python main.py --input data/train/sample_screenshot.jpg
```

### With All Flags (Recommended for Development)
```bash
.venv/bin/python main.py --input data/train/sample_screenshot.jpg --verbose --debug
```

---

## Running Tests

### Validation Tests
```bash
.venv/bin/python test_validation.py --verbose --save-results
```

### Single Image Test
```bash
.venv/bin/python test_single.py
```

---

## Command Breakdown

| Flag | Description |
|------|-------------|
| `--input` | Path to the screenshot image (required) |
| `--verbose` | Print detailed per-step timing and logs |
| `--debug` | Save intermediate debug images and display scores matrix |

---

## Output Explanation

When you run the pipeline, you'll see:

```
[PIPELINE] Detected: Active=L (pos=(0, 5)), Next=S | Time: 539ms
[ENGINE]   Best move: rotation col=3
```

**Meaning:**
- **Active piece**: L-piece currently falling
- **Position**: (row=0, col=5) on the board
- **Next piece**: S-piece coming up
- **Best move**: Best rotation index, best column = 3
- **Total time**: 539ms (including detection + engine lookahead)

### With `--debug` flag, you also get:
```
[SCORES MATRIX]
           col 0  col 1  col 2  col 3  col 4  col 5  col 6  col 7  col 8  col 9
         ----------------------------------------------------------------------
  rot 0  | -229.0 -236.0 -237.0   -inf   -inf   -inf -272.0 -256.0   -inf   -inf
  rot 1  | -217.0 -229.0 -223.0 -229.0   -inf   -inf   -inf -239.0 -235.0   -inf
  rot 2  | -219.0 -222.0 -227.0   -inf   -inf   -inf -270.0 -246.0   -inf   -inf
  rot 3  | -234.0 -234.0 -233.0 -228.0   -inf   -inf -229.0 -248.0 -219.0   -inf
```

**Explanation:**
- Each row = a unique rotation of the piece
- Each column = a column on the board (0-9)
- `-inf` = invalid placement (out of bounds or collision)
- Numbers = Dellacherie lookahead scores (higher is better)
- **Highlighted position** = the best move chosen by the engine

---

## Common Issues

### ModuleNotFoundError: No module named 'cv2'
**Solution**: Make sure you're using the venv Python:
```bash
# Make sure venv is activated or use full path:
.venv/bin/python main.py ...
```

### Command not found: python
**Solution**: Use `python3` or activate venv:
```bash
.venv/bin/python main.py ...
# OR
source .venv/bin/activate
python main.py ...
```

### FileNotFoundError for image
**Solution**: Use correct relative path or absolute path:
```bash
.venv/bin/python main.py --input ./data/train/image.jpg
# OR
.venv/bin/python main.py --input /absolute/path/to/image.jpg
```

---

## Quick Aliases (Optional)

Add to your `.bashrc` or `.zshrc` for convenience:
```bash
alias tetris='cd /home/faran/Desktop/Semester06/DIP/Project/vision-Tetris && source .venv/bin/activate'
alias tetris-run='.venv/bin/python main.py'
```

Then you can use:
```bash
tetris
tetris-run --input data/train/image.jpg --verbose
```
