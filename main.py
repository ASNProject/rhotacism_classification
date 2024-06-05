from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from test import predict_rhotacism
from record import record
import os


def button_record():
    print("Mulai Rekam")
    resultLabel.config(text="")
    remove_image()
    record()
    root.after(100, enable_button)


def enable_button():
    labelRecord.config(text="Start Detection", image=circle_photo2, compound="right")


def button_detection():
    print("Mulai Deteksi")
    result = predict_rhotacism()
    resultLabel.config(text=result)
    labelRecord.config(text="", image="")
    check_image()


def create_circle_image(diameter, color):
    image = Image.new("RGBA", (diameter, diameter), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, diameter, diameter), fill=color)
    return image


def remove_image():
    label_image.config(image="")
    label_text.config(text="")


def check_image():
    if os.path.exists("image.png") and os.stat("image.png").st_size > 0:
        img = Image.open("image.png")
        img = img.resize((450, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img
        label_image.place(x=10, y=40)
    else:
        label_text.config(text="No image available")
        label_text.place(x=10, y=40)


root = Tk()
root.title("Rhotacism Classification")
root.minsize(200, 200)
root.maxsize(500, 500)
root.geometry("480x320+50+50")

circle_image = create_circle_image(12, "red")
circle_photo = ImageTk.PhotoImage(circle_image)

circle_image2 = create_circle_image(12, "green")
circle_photo2 = ImageTk.PhotoImage(circle_image2)

titleText = Label(root, text="Rhotacism Classification", font=("Helvetica", 16))
titleText.place(x=10, y=10)

# Create button for record
buttonRecord = Button(root,
                      text="RECORD",
                      command=button_record,
                      padx=10,
                      pady=5,
                      image=circle_photo,
                      compound="right"
                      )
buttonRecord.place(x=10, y=260)

# Create button for detection
buttonDetection = Button(root,
                         text="DETECTION",
                         command=button_detection,
                         padx=4,
                         pady=4
                         )
buttonDetection.place(x=130, y=260)

labelRecord = Label(root, text="", font=("Helvetica", 12))
labelRecord.place(x=350, y=10)

resultLabel = Label(root, text="", font=("Helvetica", 12), fg="red")
resultLabel.place(x=250, y=260)

label_image = Label(root)
label_text = Label(root)

check_image()

root.mainloop()
