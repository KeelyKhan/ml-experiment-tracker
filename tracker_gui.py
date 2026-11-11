# GenAI acknowledgement: I acknowledge the use of Claude AI (Anthropic, https://claude.ai)
# to review drafts of my work and provide feedback. All code was written by me.

import tkinter as tk
from tkinter import ttk, messagebox
from tracker_core import ModelManager

class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Experiment Tracker")
        self.root.geometry("1000x600")
        self.manager = ModelManager()
        self.unsaved_changes = False

def main():
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
