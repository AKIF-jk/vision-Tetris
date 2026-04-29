# vision-based_Tetris_advisor-
A vision-based Tetris advisor that reads a game screenshot, identifies the current board state, the active piece, and the next piece preview, then recommends the optimal rotation and drop column — with a scored confidence heatmap overlaid on the board.
## What problem it solves
Tetris requires fast spatial reasoning that humans struggle with under time pressure. Our system uses image processing to perceive the board exactly as a camera would, then applies a proven evaluation function (Dellacherie, 2003) to compute the mathematically best move — including two-piece lookahead for smarter long-term decisions.
## How we are building it

<b>Step 1</b>
Preprocessing — grayscale conversion, thresholding, noise removal on the input screenshot

<b>Step 2</b>
Grid extraction — contour detection to isolate the 20×10 board and parse each cell as filled or empty

<b>Step 3</b>
Piece detection — blob analysis + Hu moments to classify both the active piece and next piece preview

<b>Step 4</b>
Rotation candidates — affine transforms generate all valid rotations; each is scored across all drop columns using Dellacherie's 6-feature evaluation function with two-piece lookahead

<b>Step 5</b>
Output — best move highlighted on board + confidence heatmap showing scores across all columns
