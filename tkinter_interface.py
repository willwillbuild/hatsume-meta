import tkinter as tk
from PIL import ImageTk, Image


def show_entry_fields():
    print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))

'''
#This creates the main window of an application
window = tk.Tk()
window.title("Join")
window.geometry("300x300")
window.configure(background='grey')
'''
master = tk.Tk()

path = "./blops_hi_flickshot.png"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))


tk.Label(master,
         text="First Name").grid(row=0)
tk.Label(master,
         text="Last Name").grid(row=1)

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
#panel = tk.Label(master, image = img)

#The Pack geometry manager packs widgets in rows or columns.
#panel.pack(side="bottom", fill="both", expand="yes")

e1 = tk.Entry(master)
e2 = tk.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

tk.Button(master,
          text='Quit',
          command=master.quit).grid(row=3,
                                    column=0,
                                    sticky=tk.W,
                                    pady=4)
tk.Button(master,
          text='Show', command=show_entry_fields).grid(row=3,
                                                       column=1,
                                                       sticky=tk.W,
                                                       pady=4)

tk.mainloop()

