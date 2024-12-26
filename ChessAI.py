# research https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2
# research https://python-chess.readthedocs.io/en/latest/
# stockfish video (research): https://www.youtube.com/watch?v=iEaU__JdI7c

from shapely.geometry import Polygon
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import cv2
import chess
import chess.engine

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
        # Runs the YOLO model on the given image and returns the results.
        results = cornerModel(image) # This is like model(image)
        # Extract bounding boxes
        bboxes = results[0].boxes
        return results, bboxes

    def draw_bounding_boxes(image, results):
        # This method draws the bounding boxes on the image using the YOLO results.
        annotated_image = results[0].plot()  # YOLO provides a method to plot bounding boxes
        return annotated_image

    def main():
        while True:
            print("Capturing image...")
            image = capture_image()
            
            results, bboxes = process_image(image) # runs the image through the corner model
            
            annotated_image = draw_bounding_boxes(image, results) # creates bounding boxes
            
            cv2.imshow("Detected Frame", annotated_image) # show the frame with detection
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the loop
                break
            
            # check if the model detected 4 bounding boxes
            if len(bboxes) == 4:
                print("Model detected 4 bounding boxes.")
                break
            else:
                print(f"Detected {len(bboxes)} bounding boxes. Retrying...")
        
        annotated_image = draw_bounding_boxes(image, results)
        print()
        
        # close all OpenCV windows
        cv2.destroyAllWindows()
        bboxes = bboxes.xywh.detach().numpy()  # Convert 'xywh' to NumPy array
        centroid = bboxes[:, :2] # this is the xyxy centroid in NumPy
        return centroid, image
    return main() # returns centroid, image

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

def calculate_centroid(box):
    """
    Calculate the centroid of a bounding box given its coordinates.
    """
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return np.array([cx, cy])

def order_points(pts):
    # order a list of 4 coordinates
    # 0: top left,
    # 1: top right
    # 2: bottom right,
    # 3: bottom left
    
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

# This function was found online.
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
    columns = "87654321" # keep this as is
    rows = "abcdefgh" # keep this as is

    fen_labels = [f"{row}{col}" for row in rows for col in columns]  # Generate FEN square labels
    fen_to_points = {}

    for idx, square in enumerate(squares):
        # get top left and bottom right coordinates of the split up chess board
        (x1, y1), (x2, y2) = square

        # make the four corners
        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)

        # map to the FEN label
        fen_to_points[fen_labels[idx]] = [top_left, top_right, bottom_left, bottom_right]

    return fen_to_points

def calculate_iou(box_1, box_2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        box_1 (list of tuples): Coordinates of the first bounding box.
        box_2 (list of tuples): Coordinates of the second bounding box.

    Returns:
        float: IoU value between 0 and 1.
    """
    # make sure the bounding boxes are closed polygons
    if box_1[0] != box_1[-1]:
        box_1.append(box_1[0])
    if box_2[0] != box_2[-1]:
        box_2.append(box_2[0])

    # create polygons
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    # check if the polygons are valid
    if not poly_1.is_valid:
        poly_1 = poly_1.buffer(0)  # fix invalid geometry
    if not poly_2.is_valid:
        poly_2 = poly_2.buffer(0)  # fix invalid geometry

    # check if union area is valid (indicating potential overlap)
    union_area = poly_1.union(poly_2).area
    if union_area > 0:
        # calculate the intersection of union
        intersection_area = poly_1.intersection(poly_2).area
        iou = intersection_area / union_area
        return iou
    else:
        # no overlap or invalid geometries
        return 0

def convert_to_coordinates(bbox): # this converts the numpy array for bounding boxes to (x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)
    return [
        (bbox[0], bbox[1]),  # Top-left
        (bbox[2], bbox[1]),  # Top-right
        (bbox[0], bbox[3]),  # Bottom-left
        (bbox[2], bbox[3])   # Bottom-right
    ]

def mapPiecesLocation(warpedImage, piecesModel):
    results = piecesModel.predict(
    source=warpedImage,
    iou=0.9, # intersection of union at least 90%
    line_width=1,
    conf=0.60, # min confidence score is 60 percent
    save_txt=False,
    save=True,
    max_det = 32) # there will be maximum 32 detections since at most 32 piece on the board

    piecesLocation = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy() # get the coordinates of bounding box
            cls = box.cls.cpu().numpy() # get the class
            confidenceScore = box.conf.cpu().numpy() # get the confidence score of the detection
            if result.names[int(cls[0])] not in piecesLocation:
                piecesLocation[result.names[int(cls[0])]] = [xyxy, confidenceScore[0]] # add to piecesLocation
            else:
                piecesLocation[result.names[int(cls[0])]].append(xyxy) # add to piecesLocation
                piecesLocation[result.names[int(cls[0])]].append(confidenceScore[0]) # add to piecesLocation
    return piecesLocation

# This function isn't even working properly I need to go back and fix it
# As long as the model predicts the right pieces this function is not needed
def findingKing(piecesLocation): # the prediction score is on the even indicies if n is the prediction boxes n+1 is its confidence score
    importantPieces = ['white-king', 'black-king', 'black-queen', 'white-queen']
    if importantPieces[0] not in piecesLocation: # this is trying to find the white king if its not in the hashmap
        if 'white-bishop' in piecesLocation:
            if len(piecesLocation['white-bishop']) / 2 > 2: # looking at possible candidates that might have an error detection
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['white-bishop']), 2):
                    if piecesLocation['white-bishop'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['white-bishop'][i]
                        lowest_confidence_index = i
                piecesLocation['white-king'] = [piecesLocation['white-bishop'][i - 1], piecesLocation['white-bishop'][i]] # added the king to the board
                del piecesLocation['white-bishop'][lowest_confidence_index-1:lowest_confidence_index+1]

        elif 'white-queen' in piecesLocation:
            if len(piecesLocation['white-queen']) / 2 > 1:
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['white-queen']), 2):
                    if piecesLocation['white-queen'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['white-queen'][i]
                        lowest_confidence_index = i
                piecesLocation['white-king'] = [piecesLocation['white-queen'][i - 1], piecesLocation['white-queen'][i]] # added the king to the board
                del piecesLocation['white-queen'][lowest_confidence_index-1:lowest_confidence_index+1]

        elif 'white-rook' in piecesLocation:
            if len(piecesLocation['white-rook']) / 2 > 2:
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['white-rook']), 2):
                    if piecesLocation['white-rook'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['white-rook'][i]
                        lowest_confidence_index = i
                piecesLocation['white-king'] = [piecesLocation['white-rook'][i - 1], piecesLocation['white-rook'][i]] # added the king to the board
                del piecesLocation['white-rook'][lowest_confidence_index-1:lowest_confidence_index+1]
        else:
            raise Exception("WHITE KING CANNOT BE LOCATED, PLEASE RETRY DETECTION")
        
    if importantPieces[1] not in piecesLocation: # this is trying to find the black king if its not in the hashmap
        if 'black-bishop' in piecesLocation: # I cannot assume it will be in the map
            if len(piecesLocation['black-bishop']) / 2 > 2: # looking at possible candidates that might have an error detection
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['black-bishop']), 2):
                    if piecesLocation['black-bishop'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['black-bishop'][i]
                        lowest_confidence_index = i
                piecesLocation['black-king'] = [piecesLocation['black-bishop'][i - 1], piecesLocation['black-bishop'][i]] # added the king to the board
                del piecesLocation['black-bishop'][lowest_confidence_index-1:lowest_confidence_index+1]

        elif 'black-queen' in piecesLocation: # I cannot assume it will be in the map
            if len(piecesLocation['black-queen']) / 2 > 1:
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['black-queen']), 2):
                    if piecesLocation['black-queen'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['black-queen'][i]
                        lowest_confidence_index = i
                piecesLocation['black-king'] = [piecesLocation['black-queen'][i - 1], piecesLocation['black-queen'][i]] # added the king to the board
                del piecesLocation['black-queen'][lowest_confidence_index-1:lowest_confidence_index+1]

        elif 'black-rook' in piecesLocation: # I cannot assume it will be in the map
            if len(piecesLocation['black-rook']) / 2 > 2:
                lowest_confidence = float("inf") 
                lowest_confidence_index = 0
                for i in range(1, len(piecesLocation['black-rook']), 2):
                    if piecesLocation['black-rook'][i] < lowest_confidence:
                        lowest_confidence = piecesLocation['black-rook'][i]
                        lowest_confidence_index = i
                piecesLocation['black-king'] = [piecesLocation['black-rook'][i - 1], piecesLocation['black-rook'][i]] # added the king to the board
                del piecesLocation['black-rook'][lowest_confidence_index-1:lowest_confidence_index+1]
        else:
            raise Exception("BLACK KING CANNOT BE LOCATED, PLEASE RETRY DETECTION")
    return piecesLocation

def halveBoundingBox(piecesLocation):
    importantPieces = ['white-king', 'black-king', 'black-queen', 'white-queen']
    
    # ensure that both kings are present, if not, find them
    if importantPieces[0] not in piecesLocation or importantPieces[1] not in piecesLocation: 
        piecesLocation = findingKing(piecesLocation) # honestly this line is not even working properly

    for piece in importantPieces:
        if piece in piecesLocation:
            box_data = piecesLocation[piece]
            
            for i in range(len(box_data)):
                box = box_data[i]
                
                # Handle NumPy array format (array([[x_min, y_min, x_max, y_max]]))
                if isinstance(box, np.ndarray):
                    # Ensure that the array has the correct shape for unpacking
                    if box.shape == (1, 4):
                        x_min, y_min, x_max, y_max = box[0]
                        mid_y = y_min + (y_max - y_min) / 2
                        box[0][1] = mid_y  # Update y_max directly CHANGE TO box[0][3]
                
                # Handle regular list format (e.g., [x_min, y_min, x_max, y_max])
                elif isinstance(box, list) and len(box) == 4:
                    x_min, y_min, x_max, y_max = box
                    mid_y = y_min + (y_max - y_min) / 2
                    box_data[i] = [x_min, mid_y, x_max, y_min] # CHANGE TO [x_min, y_min, x_max, mid_y] 
    return piecesLocation

def mapPieceToFEN(piecesLocation, fenDictonary): # at most 2084 iterations 32 max pieces * 64 board squares
    print(piecesLocation)
    fenMapping = list(fenDictonary.keys())
    piecesMapping = list(piecesLocation.keys())
    FEN_TO_PIECE = {}
    for FEN in fenMapping:
        FEN_TO_PIECE[FEN] = None

    for piece in piecesMapping: # iterating though the hashmap
        for i in range(0, len(piecesLocation[piece]), 2): # getting every
            maxConfidence = float("-inf")
            fen = None
            for FEN in fenMapping: # every single piece like a single pawn needs to go through every FEN (ERROR UNDER THIS FOR LOOP)
                if calculate_iou(fenDictonary[FEN], convert_to_coordinates(piecesLocation[piece][i][0].tolist())) > 0.5:
                    fen = FEN
                    break
                elif calculate_iou(fenDictonary[FEN], convert_to_coordinates(piecesLocation[piece][i][0].tolist())) > maxConfidence:
                    fen = FEN
                    maxConfidence = calculate_iou(fenDictonary[FEN], convert_to_coordinates(piecesLocation[piece][i][0].tolist()))
                else:
                    continue

            FEN_TO_PIECE[fen] = piece

    return FEN_TO_PIECE

def chessAI(FEN_TO_PIECE, turn):
    piece_to_stockfish = {
        "black-pawn": "p", "black-rook": "r", "black-king": "k", 
        "black-queen": "q", "black-bishop": "b", "black-knight": "n", 
        "white-pawn": "P", "white-rook": "R", "white-king": "K", 
        "white-queen": "Q", "white-bishop": "B", "white-knight": "N"
    }

    rows = "87654321" # I flipped them so that the board will be in my perspective in the output
    columns = "abcdefgh"
    
    # convert pieces to Stockfish
    for column in columns:
        for row in rows:
            fen = f"{column}{row}"
            if FEN_TO_PIECE[fen] is not None:
                FEN_TO_PIECE[fen] = piece_to_stockfish[FEN_TO_PIECE[fen]]
    
    boardMapping = ""
    # make the FEN string
    for row in rows:  # traverse rows from 8 to 1
        count = 0
        rowMapping = ""
        for column in columns:
            fen = f"{column}{row}"
            piece = FEN_TO_PIECE[fen]

            if piece is None:
                count += 1
            else:
                if count > 0 and piece is not None:
                    rowMapping += str(count)
                    rowMapping += piece
                    count = 0
                else:
                    rowMapping += piece

        if count > 0:
            rowMapping += str(count)
        
        if column is not 8:
            rowMapping += "/"

        boardMapping += rowMapping

    boardMapping = boardMapping[:-1] # remove the last "/"
    boardMapping += f" {turn}"
    print(f"FEN string: {boardMapping}")

    # create the chess board using the generated FEN string
    try:
        board = chess.Board(boardMapping[:-2]) # this removes " w" or " b"
        print(board)
    except ValueError as e:
        print(f"Error in creating the board: {e}")

    if board.is_game_over():
        if board.result() == "1-0":
            print("White Won!")
        else:
            print("Black Won!")
    else:
        engine_path = "/opt/homebrew/bin/stockfish"
        if turn == "w":
            board.turn = True # whites turn 
        else:
            board.turn = False # blacks turn 

        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            print()
            print("Thinking...")
            print()
            # get the best move
            result = engine.play(board, chess.engine.Limit(time=1.0))  # the engine is predicting with a 1 second thinking time
            print("Best move:", result.move)

            # apply the best move to the board
            board.push(result.move)
            print("Updated board:")
            print(board)

# This function is executing everything
def main(piecesModel, centroid, image, turn):
    centroids = []
    for j in range(4):
        centroids.append(centroid[j])

    ordered_boxesPoints = ordered_centroid_points(centroids)
    
    warped_image = crop_and_warp(image, ordered_boxesPoints) 

    fenDictonary = fenMatrix(infer_grid(warped_image))

    piecesLocation = mapPiecesLocation(warped_image, piecesModel)

    processedLocations = halveBoundingBox(piecesLocation)

    FEN_TO_PIECE = mapPieceToFEN(processedLocations, fenDictonary)

    chessAI(FEN_TO_PIECE, turn)

if "__main__" == __name__: # you can call the functions from the direct script if you want
    pass 
