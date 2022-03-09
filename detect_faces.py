import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

def get_image(text):
    global path
    path = ''
    if text == 'webcam':
        path = '0'
    elif text == 'video':
        path = filedialog.askopenfilenames()[0]
    else:
        path = filedialog.askopenfilenames()[0]


def image_Choice():
    root = tk.Tk()
    root.title("Face detection App")
    root.geometry("400x250")

    tk.Button(root, text="From webcam", command=lambda: get_image("webcam")).grid(row=1, column=0, padx=8, pady=70)
    tk.Button(root, text="Import video", command=lambda: get_image("video")).grid(row=1, column=1, padx=8, pady=70)
    tk.Button(root, text="Import image", command=lambda: get_image("image")).grid(row=1, column=2, padx=8, pady=70)

    tk.Button(root, text="Detect", anchor='center', command=lambda: detect(path=path)).grid(row=2, column=0)
    tk.Button(root, text="Exit", bg="#DDDDDD", anchor='center', command=root.destroy).grid(row=2, column=1)

    root.mainloop()

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def popupmsg():
    popup = tk.Tk()
    popup.resizable(0, 0)
    popup.geometry("150x80")
    popup.title("Alert")
    label = tk.Label(popup, text="Invalid input")
    label.pack(side="top", fill="x", pady=5, padx=5)
    p1 = tk.Button(popup, text="OK", bg="#DDDDDD", command=popup.destroy)
    p1.pack()

def convert_images_tk(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtK = ImageTk.PhotoImage(image)
    return imgtK

def show_images_tk(org_image, det_image, windowname):
    size, height = org_image.shape[1], org_image.shape[0]
    image1 = org_image.copy()
    image2 = det_image.copy()
    imgWin = tk.Toplevel()
    imgWin.title(windowname)
    # imgWin.geometry(f"{size*2 + 4}x{height + 20}")
    canvas1 = tk.Canvas(imgWin, width=size, height=height)
    canvas1.pack(padx=0, pady=1, side='left')
    canvas2 = tk.Canvas(imgWin, width=size, height=height)
    canvas2.pack(padx=0, pady=1, side='right')

    org_image = convert_images_tk(org_image)
    det_image = convert_images_tk(det_image)

    canvas1.create_image(0, 0, anchor='nw', image=org_image)
    canvas2.create_image(0, 0, anchor='nw', image=det_image)

    button1 = tk.Button(imgWin, text="Save", bg="#DDDDDD", command=lambda:cv2.imwrite(f'{windowname}.png', image2))
    button1.pack(side='left', anchor='center', padx=4, pady=4)
    button2 = tk.Button(imgWin, text="Exit", bg="#DDDDDD", command=imgWin.destroy)
    button2.pack(side='right', anchor='center', padx=4, pady=4)

    imgWin.mainloop()

def detect_show_image(image):
    imgCopy = image.copy()
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (255, 0, 0), 2)
    show_images_tk(image, imgCopy, "Detected face")


def detect_show_captures(cap):
    while True:
        _, image = cap.read()
        imgCopy = image.copy()
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        show_images_tk(image, imgCopy, 'Face detection in video')

def detect(path):
    image_extensions = ['.jpg', '.jpeg', '.png']
    if path == '0':
        cap = cv2.VideoCapture(0)
        detect_show_captures(cap)
    elif path.endswith('.mp4'):
        cap = cv2.VideoCapture(path)
        detect_show_captures(cap)
    elif path.endswith(tuple(image_extensions)):
        image = cv2.imread(path)
        detect_show_image(image)
    else:
        popupmsg()

if __name__ == '__main__':
    image_Choice()