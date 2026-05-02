import sys, os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.grid_parser import parse_grid
from src.piece_detector import detect_piece

if len(sys.argv) < 2:
    print('Usage: python src/debug_run_piece.py <image_path>')
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    print('Failed to load image:', img_path)
    sys.exit(2)

grid = parse_grid(img, save_debug=True, debug_prefix='test_img')
res = detect_piece(img, grid, save_debug=True, debug_prefix='test_img')
print('Detection result:', res)
