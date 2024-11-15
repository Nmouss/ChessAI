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

def main(model, image):
    # Run prediction
    results = model.predict(
        source='/Users/nabilmouss/Downloads/IMG_9122.jpeg',
        line_width=1,
        conf=0.25,
        save_txt=False,
        save=False
    )

    # Here I am extracting the bounding boxes coordinates of the corner box detection
    boxesPoints = []
    centroids = []
    for r in results: # makes 4 arrays with 4 points in each array
        for box in r.boxes:
            # Extract coordinates in (x, y) format for each corner
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Get the bounding box
            pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
            ordered_rect = order_points(pts)  # order the points (calls the method)
            boxesPoints.append(ordered_rect)

    # Here I am getting the centroid (middle) of the boxes coordinates
    for i in range(4):
        centroids.append(calculate_centroid(boxesPoints[i])[0])
    print(centroids)

    # Apply perspective transform for each detected box
    ordered_boxesPoints = [centroids[3], centroids[1], centroids[2], centroids[0]]
    warped_image = crop_and_warp(image, ordered_boxesPoints)
    
    # Here Im showing the chess board
    cv2.imshow("Warped Chessboard", warped_image)  # Display the first warped chessboard
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YOLO("/Users/nabilmouss/Desktop/ChessBoardData/ChessCornerDetection/weights/best.pt")
    image = cv2.imread("/Users/nabilmouss/Downloads/IMG_9122.jpeg")
    main(model, image)

