# research https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import cv2
# Load the model.
# model = YOLO('yolov8n.pt')

# # Training.
# results = model.train(
#     data='/Users/nabilmouss/Downloads/ChessBoardEdges/data.yaml',
#     imgsz=640,
#     epochs=100,
#     augment=True,
#     batch=16,
#     name='ChessCornerDetection',
#     project='/Users/nabilmouss/Downloads')

def ordered_centroid_points(pts):
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left
    pts = np.array(pts)
    ordered_points = np.zeros((4, 2), dtype="float32")

    # Calculate the sum and difference of each point's coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # Use sum and difference to identify the corners
    ordered_points[0] = pts[np.argmin(s)]    # Top-left: smallest sum of x + y
    ordered_points[2] = pts[np.argmax(s)]    # Bottom-right: largest sum of x + y
    ordered_points[1] = pts[np.argmin(diff)] # Top-right: smallest x - y
    ordered_points[3] = pts[np.argmax(diff)] # Bottom-left: largest x - y

    return ordered_points

def get_corner_coordinates(results):
    # Here I am extracting the bounding boxes coordinates of the corner box detection
    boxesPoints = []
    for r in results: # makes 4 arrays with 4 points in each array
        for box in r.boxes:
            # Extract coordinates in (x, y) format for each corner
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Get the bounding box
            pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
            ordered_rect = order_points(pts)  # order the points (calls the method)
            boxesPoints.append(ordered_rect)
    return boxesPoints

def calculate_centroid(box):
    """
    Calculate the centroid of a bounding box given its coordinates.
    """
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return np.array([cx, cy])

def order_points(pts):
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left
    
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def distance_between(p1, p2):
    """
    Code from https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
    """
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img):
    """Infers 64 cell grid from a square image."""
    squares = []
    side = img.shape[0]  # Corrected: Get the size of the image side directly
    side = side / 8  # Divide the side by 8 to get the grid size
    for i in range(8):
        for j in range(8):
            p1 = (int(i * side), int(j * side))  # Top left corner of a bounding box
            p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares

def draw_grid_on_chessboard(warped_img):
    """Draws an 8x8 grid on a warped chessboard image."""
    grid_img = warped_img.copy()  # Make a copy to draw the grid on
    squares = infer_grid(warped_img)

    # Draw each square as a rectangle on the grid image
    for (p1, p2) in squares:
        cv2.rectangle(grid_img, p1, p2, (0, 255, 0), 3)  # Green grid lines with thickness of 1 pixel
    
    return grid_img

def main(model, image):
    # Run prediction
    results = model.predict(
        source='/Users/nabilmouss/Downloads/IMG_9122.jpeg',
        line_width=1,
        conf=0.25,
        save_txt=False,
        save=False
    )

    boxesPoints = get_corner_coordinates(results) # there are 16, 4 per corner

    centroids = []
    
    # Here I am getting the centroid (middle) of the boxes coordinates
    for i in range(4):
        centroids.append(calculate_centroid(boxesPoints[i])[0])

    # Apply perspective transform for each detected box
    ordered_boxesPoints = ordered_centroid_points(centroids)

    warped_image = crop_and_warp(image, ordered_boxesPoints) # I NEED THIS TO DO THE CROP AND WARP FOR MY NEXT STEP

    split_up_chessboard = draw_grid_on_chessboard(warped_image)

    # Here Im showing the chess board
    cv2.imshow("Warped Chessboard", split_up_chessboard)  # Display the first warped chessboard
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    model = YOLO("/Users/nabilmouss/Desktop/ChessBoardData/ChessCornerDetection/weights/best.pt")
    image = cv2.imread("/Users/nabilmouss/Downloads/IMG_9122.jpeg")
    main(model, image)


