from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from test import predict_rhotacism, prediction_mlp
from record import record
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Variabel global untuk menyimpan data recording
frames = []
sample_rate = 44100
chunk_size = 4096


def button_record():
    print("Mulai Rekam")
    resultLabel.config(text="")
    textRhotacism.config(text="")
    textNormal.config(text="")
    enable_text()
    remove_image()
    global frames
    frames = []  # Reset frames
    threading.Thread(target=record, args=(callback,), daemon=True).start()  # Run record in a thread
    root.after(12000, enable_button)


def callback(data):
    global frames
    frames.append(data)  # Simpan data ke dalam frames
    # Update waveform
    update_waveform(np.frombuffer(data, dtype=np.int16))


def enable_button():
    labelRecord.config(text="Recording Successfully\nStart Detection",
                       anchor="w", justify="left")


def enable_text():
    labelWords.config(text='"Laler Menclok Pager"')


def button_detection():
    print("Mulai Deteksi")
    # status, rhotacism, normal = predict_rhotacism()
    status, rhotacism, normal = prediction_mlp()
    labelRecord.config(text="", image="")
    labelWords.config(text="")
    textRhotacism.config(text=f"Rhotacism : {rhotacism * 100:.2f}%")
    textNormal.config(text=f"Normal      : {normal * 100:.2f}%")
    if status == "Normal":
        resultLabel.config(text=f"'{status}'", fg="green")
    else:
        resultLabel.config(text=f"'{status}'", fg="red")

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
        img = img.resize((500, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        label_image.config(image=img)
        label_image.image = img
        label_image.place(x=500, y=350)
    else:
        label_text.config(text="No image available")
        label_image.place(x=500, y=350)


def update_waveform(data):
    ax.clear()
    ax.plot(data, color='blue')
    ax.set_ylim([-32768, 32767])  # Sesuaikan dengan rentang data 16-bit
    ax.set_title("Real-time Waveform")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    canvas.draw()


# Inisialisasi GUI
root = Tk()
root.title("Rhotacism Detection")
root.geometry("1024x600")  # Ubah ukuran jendela utama jika perlu

circle_image = create_circle_image(12, "red")
circle_photo = ImageTk.PhotoImage(circle_image)

circle_image2 = create_circle_image(12, "green")
circle_photo2 = ImageTk.PhotoImage(circle_image2)

titleText = Label(root, text="RHOTACISM DETECTION", font=("Helvetica", 24))
titleText.pack(side=TOP, pady=10)

# Create button for record
buttonRecord = Button(root,
                      text="RECORD",
                      command=button_record,
                      padx=10,
                      pady=5,
                      image=circle_photo,
                      compound="right"
                      )
buttonRecord.place(x=10, y=280)

# Create button for detection
buttonDetection = Button(root,
                         text="DETECTION",
                         command=button_detection,
                         padx=4,
                         pady=4
                         )
buttonDetection.place(x=130, y=280)

labelRecord = Label(root, text="", font=("Helvetica", 12))
labelRecord.place(x=250, y=280)

labelWords = Label(root, text="", font=("Helvetica", 24))
labelWords.place(x=100, y=150)

text1 = Label(root, text="When recording, please say the sentence below:", font=("Helvetica", 12))
text1.place(x=10, y=60)

text2 = Label(root, text="Result :", font=("Helvetica", 16))
text2.place(x=10, y=320)

text3 = Label(root, text="Waveform Result:", font=("Helvetica", 16))
text3.place(x=500, y=320)

textRhotacism = Label(root, text="Rhotacism :", font=("Helvetica", 14))
textRhotacism.place(x=10, y=350)

textNormal = Label(root, text="Normal      :", font=("Helvetica", 14))
textNormal.place(x=10, y=370)

resultLabel = Label(root, text="", font=("Helvetica", 32))
resultLabel.place(x=100, y=430)

label_image = Label(root)
label_text = Label(root)

# Matplotlib setup dengan ukuran yang lebih kecil
fig, ax = plt.subplots(figsize=(4, 3), dpi=25)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().place(x=500, y=60, width=500, height=200)
ax.set_title("Waveform Recording")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

root.mainloop()
