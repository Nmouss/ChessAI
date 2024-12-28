# What is this?
This project is a project that combined all of my knowledge in computer vision, OOP, machine learning, software engineering, and reading documentation. I also learned a lot between GPUs, TPUs and CPUs when I was training my chess pieces model on a GTX 1080Ti.

# How can I use this?
You will need:

- External webcam

## Packages (For mac users)

```
pip install ultralytics
pip install python-chess
pip install shapely
pip install opencv-python
pip install numpy
brew stockfish

run this command to find stockfish in files: which stockfish
```

The ChessAI script has the algorithms that processes the images finds where the chess board corners are and find where the chess pieces are. Since I cannot attach the model here you can contact me or I will link a google drive for the models but I haven't decided what I want to do yet. Run the main script with the model detecting corners and chess pieces.

# Goal
My goal is to create glasses that a player can wear that has buttons so that they can either speak to the mic or click buttons so that the AI knows if its whites or blacks turn. So this project is not done but the main AI algorithm and computer vision parts are all done!

# Example

Its your turn to move as white what would you do? Ask AI!

Step 1) Run the script in main.py and type whos turn it is (I will hopefully have buttons in the future or even voice command using google open source voice interpreter!)

Step 2) Get the entire board in frame
![final_image](https://github.com/user-attachments/assets/a58169df-10fb-4768-a709-a25821a714b7)

Step 3) Get the 4 corner detections
![annoted_corners](https://github.com/user-attachments/assets/8033eba9-e958-4521-9270-90b5a8edb26c)

Step 4) Make sure the chess pieces are correct (the biggest issues are the king and queen). Heres whats happening behind the scene.
![image0](https://github.com/user-attachments/assets/b6303c0e-6e47-43bc-bc0a-59da4b0e8a8a)

Step 5) Look at the output that is given (Again I want this to be over voice command in the future!)
Inital board
. . . r . . . r
. p k . . p p p
p . P . p n . .
. . b p . . . .
. P . . . . . .
. . . P B . . .
. q . . P P P P
R N . . K B . R

Thinking...

Best move: b4c5
Updated board:
. . . r . . . r
. p k . . p p p
p . P . p n . .
. . P p . . . .
. . . . . . . .
. . . P B . . .
. q . . P P P P
R N . . K B . R

Step 6) Move your piece in real life.
