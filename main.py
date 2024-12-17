# research https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
from shapely import Polygon
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import cv2
import torch
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

def videoCap(cornerModel):
    def capture_image(camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise Exception("Could not open camera.")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Failed to capture image.")
        return frame

    def process_image(image):
        """
        Runs the YOLO model on the given image and returns the results.
        """
        results = cornerModel(image)
        # Extract bounding boxes
        bboxes = results[0].boxes
        return results, bboxes

    def draw_bounding_boxes(image, results):
        """
        Draws the bounding boxes on the image using the YOLO results.
        """
        annotated_image = results[0].plot()  # YOLO provides a method to plot bounding boxes
        return annotated_image

    def main():
        while True:
            # Capture an image
            print("Capturing image...")
            image = capture_image()
            
            # Run the image through the model
            results, bboxes = process_image(image)
            
            # Draw bounding boxes
            annotated_image = draw_bounding_boxes(image, results)
            
            # Show the frame
            cv2.imshow("Detected Frame", annotated_image)
            
            # Wait for a key press or 1ms delay
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
                break
            
            # Check if the model detected 4 bounding boxes
            if len(bboxes) == 4:
                print("Model detected 4 bounding boxes.")
                break
            else:
                print(f"Detected {len(bboxes)} bounding boxes. Retrying...")
        
        annotated_image = draw_bounding_boxes(image, results)
        # Save or use the final image
        cv2.imwrite("final_image.jpg", image)
        print("Final image saved as 'final_image.jpg'.")
        print()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        return bboxes, image
    return main()

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

def get_corner_coordinates(boxes):
    # Here I am extracting the bounding boxes coordinates of the corner box detection
    boxesPoints = []
    for i in range(4):
        for box in boxes:
            print("BOX", box)
            # # Extract coordinates in (x, y) format for each corner
            # x_min, y_min, x_max, y_max = box[i]  # Get the bounding box
            # pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
            # ordered_rect = order_points(pts)  # order the points (calls the method)
            # boxesPoints.append(ordered_rect)
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

# This function was found online.
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
    squares = infer_grid(warped_img) # THIS IS HOW I GET THE SQUARE COORDINATES (split evenly)

    # Draw each square as a rectangle on the grid image
    for (p1, p2) in squares:
        cv2.rectangle(grid_img, p1, p2, (0, 255, 0), 3)  # Green grid lines with thickness of 1 pixel

    return grid_img

def fenMatrix(squares):
    columns = "abcdefgh"  # Chess columns
    rows = "87654321"     # Chess rows (8 at top, 1 at bottom)
    
    fen_labels = [f"{col}{row}" for row in rows for col in columns] # this creates a list of all FEN square labels
    
    fen_to_points = {}
    for i, fen_label in enumerate(fen_labels):
        
        x1, y1 = squares[i][0] # top left points
        x2, y2 = squares[i][1] # bottom right
        
        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)
        
        fen_to_points[fen_label] = [top_left, top_right, bottom_left, bottom_right] # maps to the dictionary
    return fen_to_points # returning a dictionary

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def calculate_chess_pieces_location():
    piecesLocation = {}


def main(piecesModel, results, image):
    # Run prediction

    #boxesPoints = get_corner_coordinates(results) # there are 16, 4 per corner (I DONT NEED THIS BUT KEEP JUST IN CASE)

    centroids = []
    
    # Here I am getting the centroid (middle) of the boxes coordinates
    for j in range(4):
        centroids.append(results[j])

    # Apply perspective transform for each detected box
    ordered_boxesPoints = ordered_centroid_points(centroids)
    print(ordered_boxesPoints)
    warped_image = crop_and_warp(image, ordered_boxesPoints) # I NEED THIS TO DO THE CROP AND WARP FOR MY NEXT STEP

    split_up_chessboard = draw_grid_on_chessboard(warped_image)
    
    print(fenMatrix(infer_grid(warped_image)))
    #print(calculate_iou(fenMatrix(infer_grid(warped_image))["a8"], fenMatrix(infer_grid(warped_image))["a6"]))

    # Here Im showing the chess board
    cv2.imwrite("warpedChess_image.jpg", warped_image)
    cv2.imshow("Warped Chessboard", split_up_chessboard)  # Display the first warped chessboard
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    piecesModel = YOLO("/PATH/TO/YOUR/ChessPieces-Detection/best.pt") # this needs to be retrained
    cornerModel = YOLO("/PATH/TO/YOUR/ChessCorner-Detection/best.pt")
    boxes, image = videoCap(cornerModel)
    xywh = boxes.xywh.detach().numpy()  # Convert 'xywh' to NumPy array
    centroids = xywh[:, :2]
    main(piecesModel, centroids, image)




