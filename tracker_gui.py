# GenAI acknowledgement: I acknowledge the use of Claude AI (Anthropic, https://claude.ai)
# to review drafts of my work and provide feedback. All code was written by me.

import tkinter as tk
from tkinter import ttk, messagebox
from tracker_core import Model, ModelManager

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
        commands = {"Add": self.open_run_form, "Delete": self.delete_selected}
        for text in ("Add", "Edit", "Delete", "Save", "Load"):
            ttk.Button(left, text=text, command=commands.get(text)).pack(fill="x", pady=3)

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
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=90, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        self.status = ttk.Label(self.root, text="Ready")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.refresh_table(list(self.manager.get_models().items()))

    def open_run_form(self):
        window = tk.Toplevel(self.root)
        window.title("Add a Run")

        labels = ("Model Name", "Training Accuracy", "Training Loss", "Validation Accuracy",
                  "Validation Loss", "Epochs", "Batch Size", "Learning Rate")
        entries = []
        for row, text in enumerate(labels):
            ttk.Label(window, text=text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
            entry = ttk.Entry(window)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entries.append(entry)

        def save():
            try:
                model_name = entries[0].get()
                training_accuracy = float(entries[1].get())
                training_loss = float(entries[2].get())
                validation_accuracy = float(entries[3].get())
                validation_loss = float(entries[4].get())
                epochs = int(entries[5].get())
                batch_size = int(entries[6].get())
                learning_rate = float(entries[7].get())
                run = Model(model_name, training_accuracy, training_loss, validation_accuracy,
                            validation_loss, epochs, batch_size, learning_rate)
            except ValueError as e:
                messagebox.showerror("Invalid input", f"Could not add this run: {e}", parent=window)
                return
            run_id = self.manager.add_model(run)
            self.refresh_table(list(self.manager.get_models().items()))
            self.unsaved_changes = True
            self.status.config(text=f"Run added with ID {run_id}")
            window.destroy()

        ttk.Button(window, text="Save", command=save).grid(row=len(labels), column=0, columnspan=2, pady=10)

    def delete_selected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showerror("No selection", "Please select a run to delete.")
            return
        values = self.tree.item(selected[0], "values")
        run_id = int(values[0])
        if not messagebox.askyesno("Delete run", f"Delete run {run_id}?"):
            return
        self.manager.delete_model(run_id)
        self.refresh_table(list(self.manager.get_models().items()))
        self.unsaved_changes = True
        self.status.config(text=f"Run {run_id} deleted")

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
