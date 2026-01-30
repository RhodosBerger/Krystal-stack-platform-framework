import tkinter as tk
from tkinter import ttk, messagebox

def test_gui():
    root = tk.Tk()
    root.title("Test GUI - KrystalVino Setup")
    root.geometry("400x300")
    
    label = ttk.Label(root, text="KrystalVino Framework Setup Wizard\nby Dušan Kopecký", 
                     font=("Arial", 12, "bold"))
    label.pack(pady=20)
    
    def on_click():
        messagebox.showinfo("Success", "GUI is working correctly!")
        root.destroy()
    
    button = ttk.Button(root, text="Test Setup Wizard", command=on_click)
    button.pack(pady=20)
    
    root.mainloop()
    print("GUI test completed successfully!")

if __name__ == "__main__":
    test_gui()