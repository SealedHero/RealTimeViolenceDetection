import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

sys.path.append(r'Scripts\violenceDetect')
from violenceConfidence import predict_video
from ViolenceDetectFrames import predict_frames
SEQUENCE_LENGTH = 16
inputFile = ''

def select_file():
    global inputFile  # Declare inputFile as a global variable
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
    if file_path:
        inputFile = file_path
        print(inputFile)

def function_1():
    predict_video(inputFile, SEQUENCE_LENGTH)

def function_2():
    predict_frames(inputFile,SEQUENCE_LENGTH)

def main():
    root = tk.Tk()
    root.title("Choose Function")

    # Define a common style for the buttons
    button_style = {"padx": 20, "pady": 10, "bg": "lightblue", "fg": "black", "font": ("Arial", 12)}

    button_1 = tk.Button(root, text="Confidence", command=function_1, **button_style)
    button_1.pack()

    button_2 = tk.Button(root, text="FullVideo", command=function_2, **button_style)
    button_2.pack()

    button_3 = tk.Button(root, text="Select File", command=select_file, **button_style)
    button_3.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
