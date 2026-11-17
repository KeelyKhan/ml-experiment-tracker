# GenAI acknowledgement: I acknowledge the use of Claude AI (Anthropic, https://claude.ai)
# to review drafts of my work and provide feedback. All code was written by me.

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
        commands = {"Add": self.open_run_form, "Edit": self.edit_selected, "Delete": self.delete_selected,
                    "Save": self.save_runs, "Load": self.load_runs}
        for text in ("Add", "Edit", "Delete", "Save", "Load"):
            ttk.Button(left, text=text, command=commands.get(text)).pack(fill="x", pady=3)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=1, sticky="nsew")

        runs_frame = ttk.Frame(self.notebook)
        self.notebook.add(runs_frame, text="Runs")

        columns = ("ID", "Name", "Train Acc", "Val Acc", "Train Loss", "Val Loss", "Epochs", "Batch", "LR")
        controls = ttk.Frame(runs_frame)
        controls.pack(side="top", fill="x", pady=5)

        ttk.Label(controls, text="Sort by:").pack(side="left", padx=(5, 2))
        self.sort_box = ttk.Combobox(controls, values=("Validation accuracy", "Validation loss"), state="readonly", width=18)
        self.sort_box.current(0)
        self.sort_box.pack(side="left", padx=2)
        ttk.Button(controls, text="Sort", command=self.sort_table).pack(side="left", padx=(2, 15))

        ttk.Label(controls, text="Filter:").pack(side="left", padx=2)
        self.filter_box = ttk.Combobox(controls, values=("Accuracy ≥", "Loss ≤", "Name contains"), state="readonly", width=14)
        self.filter_box.current(0)
        self.filter_box.pack(side="left", padx=2)
        self.filter_entry = ttk.Entry(controls, width=15)
        self.filter_entry.pack(side="left", padx=2)
        ttk.Button(controls, text="Filter", command=self.filter_table).pack(side="left", padx=2)
        ttk.Button(controls, text="Show all", command=self.show_all).pack(side="left", padx=(15, 5))

        scrollbar = ttk.Scrollbar(runs_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self.tree = ttk.Treeview(runs_frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.tree.yview)
        for col in columns:
            self.tree.heading(col, text=col, anchor="center")
            self.tree.column(col, width=90, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)

        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")

        stats_controls = ttk.Frame(stats_frame)
        stats_controls.pack(side="top", fill="x", pady=5)
        ttk.Label(stats_controls, text="Metric:").pack(side="left", padx=(5, 2))
        self.stats_box = ttk.Combobox(stats_controls, values=("Validation accuracy", "Validation loss"), state="readonly", width=18)
        self.stats_box.current(0)
        self.stats_box.pack(side="left", padx=2)
        ttk.Button(stats_controls, text="Compute", command=self.show_statistics).pack(side="left", padx=2)

        self.stats_label = ttk.Label(stats_frame, text="", justify="left")
        self.stats_label.pack(side="top", anchor="w", padx=10, pady=10)

        self.status = ttk.Label(self.root, text="Ready")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.refresh_table(list(self.manager.get_models().items()))

    def open_run_form(self, run_id=None):
        window = tk.Toplevel(self.root)
        window.title("Add a Run" if run_id is None else "Edit Run")

        labels = ("Model Name", "Training Accuracy", "Training Loss", "Validation Accuracy",
                  "Validation Loss", "Epochs", "Batch Size", "Learning Rate")
        entries = []
        for row, text in enumerate(labels):
            ttk.Label(window, text=text).grid(row=row, column=0, padx=10, pady=5, sticky="w")
            entry = ttk.Entry(window)
            entry.grid(row=row, column=1, padx=10, pady=5)
            entries.append(entry)

        if run_id is not None:
            existing = self.manager.get_models()[run_id]
            values = (existing.model_name, existing.training_accuracy, existing.training_loss,
                      existing.validation_accuracy, existing.validation_loss, existing.epochs,
                      existing.batch_size, existing.learning_rate)
            for i, value in enumerate(values):
                entries[i].insert(0, value)

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
                messagebox.showerror("Invalid input", f"Could not save this run: {e}", parent=window)
                return
            if run_id is None:
                new_id = self.manager.add_model(run)
                message = f"Run added with ID {new_id}"
            else:
                self.manager.update_model(run_id, run)
                message = f"Run {run_id} updated"
            self.refresh_table(list(self.manager.get_models().items()))
            self.unsaved_changes = True
            self.status.config(text=message)
            window.destroy()

        ttk.Button(window, text="Save", command=save).grid(row=len(labels), column=0, columnspan=2, pady=10)

    def edit_selected(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showerror("No selection", "Please select a run to edit.")
            return
        values = self.tree.item(selected[0], "values")
        run_id = int(values[0])
        self.open_run_form(run_id)

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

    def save_runs(self):
        if not self.manager.get_models():
            messagebox.showerror("No runs", "There are no runs to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path:
            return
        try:
            if path.endswith(".csv"):
                self.manager.save_csv(path)
            else:
                self.manager.save_json(path)
        except OSError:
            messagebox.showerror("Save failed", "Could not save the file.")
            return
        self.unsaved_changes = False
        self.status.config(text=f"Saved {len(self.manager.get_models())} runs")

    def load_runs(self):
        if self.manager.get_models():
            if not messagebox.askyesno("Load runs", "Loading will replace your current runs. Continue?"):
                return
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("CSV", "*.csv")])
        if not path:
            return
        try:
            if path.endswith(".csv"):
                self.manager.load_csv(path)
            else:
                self.manager.load_json(path)
        except FileNotFoundError:
            messagebox.showerror("Load failed", "File not found.")
            return
        except (ValueError, KeyError, TypeError) as e:
            messagebox.showerror("Load failed", f"Could not load file: {e}")
            return
        self.refresh_table(list(self.manager.get_models().items()))
        self.unsaved_changes = True
        self.status.config(text=f"Loaded {len(self.manager.get_models())} runs")

    def sort_table(self):
        metrics = {"Validation accuracy": "validation_accuracy", "Validation loss": "validation_loss"}
        metric = metrics[self.sort_box.get()]
        try:
            pairs = self.manager.sort_runs(metric)
        except ValueError as e:
            messagebox.showerror("Sort failed", str(e))
            return
        self.refresh_table(pairs)
        self.status.config(text=f"Sorted by {self.sort_box.get().lower()}")

    def filter_table(self):
        choice = self.filter_box.get()
        value = self.filter_entry.get()
        try:
            if choice == "Accuracy ≥":
                pairs = self.manager.filter_by_accuracy(float(value))
            elif choice == "Loss ≤":
                pairs = self.manager.filter_by_loss(float(value))
            else:
                pairs = self.manager.filter_by_name(value)
        except ValueError as e:
            messagebox.showerror("Invalid filter", f"Could not filter: {e}")
            return
        self.refresh_table(pairs)
        self.status.config(text=f"Showing {len(pairs)} runs")

    def show_all(self):
        self.refresh_table(list(self.manager.get_models().items()))
        self.status.config(text="Showing all runs")

    def show_statistics(self):
        if not self.manager.get_models():
            messagebox.showerror("No runs", "There are no runs to compute statistics on.")
            return
        label = self.stats_box.get()
        metric = "validation_accuracy" if label == "Validation accuracy" else "validation_loss"
        try:
            results = self.manager.get_statistics(metric)
        except ValueError as e:
            messagebox.showerror("Statistics failed", str(e))
            return
        best_id = results["best_run_id"]
        best_name = self.manager.get_models()[best_id].model_name
        self.stats_label.config(text=f"""Statistics of {label}:
Mean: {results['mean']:.4f}
Variance: {results['variance']:.4f}
Standard Deviation: {results['std_dev']:.4f}
Best Run ID: {best_id} ({best_name})""")
        self.status.config(text=f"Statistics computed for {label.lower()}")

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
