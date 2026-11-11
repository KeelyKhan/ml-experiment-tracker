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
        self.build_layout()

    def build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="ns", padx=10, pady=(30, 0))
        for text in ("Add", "Edit", "Delete", "Save", "Load"):
            ttk.Button(left, text=text).pack(fill="x", pady=3)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=1, sticky="nsew")

        runs_frame = ttk.Frame(notebook)
        notebook.add(runs_frame, text="Runs")

        columns = ("ID", "Name", "Train Acc", "Val Acc", "Train Loss", "Val Loss", "Epochs", "Batch", "LR")
        scrollbar = ttk.Scrollbar(runs_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.tree = ttk.Treeview(runs_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.tree.yview)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=90)
        self.tree.pack(side="left", fill="both", expand=True)

        self.status = ttk.Label(self.root, text="Ready")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.refresh_table(list(self.manager.get_models().items()))

    def refresh_table(self, pairs):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for run_id, run in pairs:
            self.tree.insert("", "end", values=(
                run_id,
                run.model_name,
                f"{run.training_accuracy:.2f}",
                f"{run.validation_accuracy:.2f}",
                f"{run.training_loss:.4f}",
                f"{run.validation_loss:.4f}",
                run.epochs,
                run.batch_size,
                f"{run.learning_rate:g}",
            ))

def main():
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
