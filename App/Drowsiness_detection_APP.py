from drowsiness_detection import *
from lib import *
global filename

def addFile():
    filename = filedialog.askopenfilename(initialdir="./", title="Select file",
                                          filetypes=(("JPEG", "*.jpg"), 
                                                     ("PNG", "*.png")))
    return filename


# initial root
root = tk.Tk()
root.title('Drowsiness Detection')
root.geometry("800x400")

bg = tk.PhotoImage(file="./background.png")
  
# Show image using label
label1 = Label( root, image = bg)
label1.place(x = 0,y = 0)
webcam = tk.Button(root, text="Nhận dạng dùng camera", padx=10, pady=10, fg="black", bg="#ffffff",
                     command=lambda: video())
withimage = tk.Button(root, text="Nhận dạng hình ảnh", padx=10, pady=10, fg="black", bg="#ffffff",
                     command=lambda: image(addFile()))

webcam.place(x = 100,y = 250)
withimage.place(x = 550,y = 250)

# start app
root.mainloop()