from tkinter import *
from tkinter import messagebox
from waiting import wait
t = Tk()
t.geometry('400x400')
var = ''
var = messagebox.showinfo("Notification", "If Your Name Appears \n Press q")

def is_something_ready(var):
    if var == 'ok':
        return True
    return False


# wait for something to be ready
something = var

wait(lambda: is_something_ready(something), timeout=120, waiting_for="something to be ready")

# this code will only execute after "something" is ready
print("Done")

f1 = Frame()
f1.place(x=0,y=0,width=400,height=400)
def login():
	f2 = Frame()
	f2.place(x=0,y=0,width=400,height=400)
	e1 = Entry(f2)
	e1.pack()
	e2 = Entry(f2)
	e2.pack()
	b2 = Button(f2,text='Login')
	b2.pack()	



b1 = Button(f1,text='Click', command=login)
b1.pack()

t.mainloop()