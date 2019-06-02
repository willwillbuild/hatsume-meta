from tkinter import *
from PIL import ImageTk,Image

root = Tk()
canvas = Canvas(root, width=1000, height=500)
canvas.pack()
img = ImageTk.PhotoImage(Image.open("./blops_med_flickshot.png").resize((900,463)))
canvas.create_image(40, 10, anchor=NW, image=img)

w = Label(root, text="Excitement Index")
w.pack()
e1 = Entry(root)

e1.pack(pady=10)

root.mainloop()