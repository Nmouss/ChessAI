import ChessAI as Chess
from ultralytics import YOLO
print("type 'w', 'b' or 'white', 'black'") # of course this will be replaced with buttons when i can get glasses that have cameras
turn = input("White or Black's turn to go? ").lower()
if turn == 'w' or turn == 'white':
    turn = 'w'
else:
    turn = 'b'

piecesModel = YOLO("/Users/nabilmouss/Desktop/ChessBoardData/ChessPieces-Detection/best.pt")
cornerModel = YOLO("/Users/nabilmouss/Desktop/ChessBoardData/ChessCorner-Detection/best.pt")
centroid, images = Chess.videoCap(cornerModel)
Chess.main(piecesModel, centroid, images, turn)
