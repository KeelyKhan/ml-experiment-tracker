# GenAI acknowledgement: I acknowledge the use of Claude AI (Anthropic, https://claude.ai)
# to review drafts of my work and provide feedback. All code was written by me.

from tracker_core import Model,ModelManager

def print_menu():
    menu = """
        1. Show runs
        2. Add a run
        3. Update a run
        4. Delete a run
        5. Sort runs
        6. Filter / search runs
        7. Show statistics
        8. Get recommendations
        9. Save runs
        10. Load runs
        0. Quit\n
    """
    print(menu)

def print_runs(pairs):
    if not pairs:
        print("\nNo runs to show")
        return 
    print(f"{'ID':<4}{'Name':<14}{'Train Acc':<11}{'Val Acc':<11}{'Train Loss':<11}{'Val Loss':<11}{'Epochs':<8}{'Batch':<8}{'LR':<10}")
    for run_id,run in pairs:
        print(f"{run_id:<4}{run.model_name:<14}{run.training_accuracy:<11.2f}{run.validation_accuracy:<11.2f}{run.training_loss:<11.4f}{run.validation_loss:<11.4f}{run.epochs:<8}{run.batch_size:<8}{run.learning_rate:<10g}")

def ask_float(prompt):
    while True:
        text = input(prompt)
        try:
            result = float(text)
            return result
        except ValueError:
            print("Please enter a valid number")

def ask_int(prompt):
    while True:
        text = input(prompt)
        try:
            result = int(text)
            return result
        except ValueError:
            print("Please enter a whole number")

def ask_text(prompt):
    while True:
        text = input(prompt).strip()
        if not text:
            print("This field cannot be empty")
        else:
            return text
        
def ask_run_details():
    model_name = ask_text("Model Name: ")
    training_accuracy = ask_float("Training Accuracy: ")
    training_loss = ask_float("Training Loss: ")
    validation_accuracy = ask_float("Validation Accuracy: ")
    validation_loss = ask_float("Validation Loss: ")
    epochs = ask_int("No of Epochs: ")
    batch_size = ask_int("Batch Size: ")
    learning_rate = ask_float("Learning Rate: ")
    return Model(model_name,training_accuracy,training_loss,validation_accuracy,validation_loss,epochs,batch_size,learning_rate)

def add_run(manager):
    try:
        run = ask_run_details()
        run_id = manager.add_model(run)
        print(f"Run added with ID {run_id}")
        return True
    except ValueError as e:
        print(f"Could not add this run: {e}")
        return False

def ask_yes_no(prompt):
    while True:
        text = input(prompt).strip().lower()
        if text == "y":
            return True
        elif text == "n":
            return False
        else:
            print("Please enter a valid option - Y for Yes, N for No")

def delete_run(manager):
    if not manager.get_models():
        print("No runs to delete")
        return False
    print("\nChoose a run ID to delete: ")
    print_runs(list(manager.get_models().items()))
    run_id = ask_int("\nEnter run ID: ")
    decision = ask_yes_no(f"Delete run {run_id}? (y/n): ")
    if not decision:
        print("\nCancelled")
        return False
    try:
        manager.delete_model(run_id)
        print(f"Run {run_id} deleted successfully")
        return True
    except KeyError:
        print(f"No run with ID {run_id} exists")
        return False

def update_run(manager):
    if not manager.get_models():
        print("No runs to update")
        return False
    print("\nChoose a run ID to update: ")
    print_runs(list(manager.get_models().items()))
    run_id = ask_int("\nEnter run ID: ")
    if run_id not in manager.get_models().keys():
        print(f"No run with ID {run_id}")
        return False
    try:
        run = ask_run_details()
        manager.update_model(run_id,run)
        print(f"Run {run_id} updated")
        return True
    except ValueError as e:
        print(f"Could not update this run: {e}")
        return False

def sort_runs_menu(manager):
    if not manager.get_models():
        print("No runs to sort")
        return
    message = """
\nHow do you want to sort your runs
  1. Validation accuracy (high to low)
  2. Validation loss (low to high)
"""
    print(message)
    choice = input("Choice 1 or 2: ").strip()
    if choice == "1":
        sorted_runs = manager.sort_runs("validation_accuracy")
        print_runs(sorted_runs)
    elif choice == "2":
        sorted_runs = manager.sort_runs("validation_loss")
        print_runs(sorted_runs)
    else:
        print("Please enter a valid choice number")

def filter_runs_menu(manager):
    if not manager.get_models():
        print("No runs to filter")
        return
    message = """
Choose an option for filter: 
1. Validation Accuracy
2. Validation Loss
3. Model Name
"""  
    print(message)
    choice = input("\nEnter your choice: ").strip()
    if choice == "1":
        val_acc = ask_float("\nMinimum validation accuracy (0-100): ")
        try:
            result = manager.filter_by_accuracy(val_acc)
            print_runs(result)
        except ValueError as e:
            print(f"Enter a valid threshold: {e}")
    elif choice == "2":
        val_loss = ask_float("\nMaximum validation loss: ")
        try:
            result = manager.filter_by_loss(val_loss)
            print_runs(result)
        except ValueError as e:
            print(f"Enter a valid threshold: {e}")
    elif choice == "3":
        model = ask_text("\nEnter Model Name: ")
        try:
            result = manager.filter_by_name(model)
            print_runs(result)
        except ValueError as e:
            print(f"Enter a valid model name: {e}")
    else:
        print("Please enter a valid choice")

def show_statistics_menu(manager):
    if not manager.get_models():
        print("No runs exist to compute statistics")
        return
    message = """
Choose a metric to compute stats:
1. Validation Accuracy
2. Validation Loss
""" 
    print(message)
    choice = input("Enter your choice: ").strip()
    if choice == "1":
        metric = "validation_accuracy"
        label = "Validation Accuracy"
    elif choice == "2":
        metric = "validation_loss"
        label = "Validation Loss"
    else:
        print("Please enter a valid option")
        return
    
    results = manager.get_statistics(metric)
    best_id = results['best_run_id']
    best_name = manager.get_models()[best_id].model_name
    stats = f"""
Statistics of {label}:
Mean: {results['mean']:.4f}
Variance: {results['variance']:.4f}
Standard Deviation: {results['std_dev']:.4f}
Best Run ID: {best_id} ({best_name})
"""
    print(stats)

def show_recommendations_menu(manager):
    if not manager.get_models():
        print("No runs exist to form recommendations")
        return
    print("Choose a run to show recommendations for: ")
    print_runs(list(manager.get_models().items()))
    run_id = ask_int("\nEnter Run ID: ")
    try:
        results = manager.recommend(run_id)
        print(f"Recommendations for Run {run_id} ({manager.get_models()[run_id].model_name})\n")
        for result in results:
            print("Issue: ",result["issue"])
            print("\nSuggestions: ",result["suggestion"])
    except KeyError:
        print(f"No run with ID {run_id} exists")
        
def save_runs_menu(manager):
    if not manager.get_models():
        print("No runs to save")
        return False
    message = """
How do you want to save your runs: 
1. JSON
2. CSV
"""
    print(message)
    save_option = input("\nEnter your choice: ").strip()
    try:
        if save_option == "1":
            path = ask_text("\nFile Name: ")
            manager.save_json(path)
        elif save_option == "2":
            path = ask_text("\nFile Name: ")
            manager.save_csv(path)
        else:
            print("Enter a valid option")
            return False
        print(f"\nSaved {len(manager.get_models())} runs")
        return True
    except OSError:
        print("Could not save the file")
        return False

def load_runs_menu(manager):
    message = """
Choose the loading format:
1. JSON
2. CSV
"""
    if manager.get_models():
       choice = ask_yes_no("Loading will replace your current runs. Continue? (y/n): ")
       if not choice:
           print("Cancelled")
           return False
    print(message)
    load_choice = input("\nEnter your choice: ").strip()
    if load_choice not in ("1","2"):
        print("Invalid Choice")
        return False
    path = ask_text("\nFile Name: ")
    try:
        if load_choice == "1":
            manager.load_json(path)
        else:
            manager.load_csv(path)
        print(f"Loaded {len(manager.get_models())} runs")
        return True
    except FileNotFoundError:
        print("File Not Found")
    except (ValueError, KeyError, TypeError) as e:
        print(f"Could not load file: {e}")
    return False


def main():
    manager = ModelManager()
    unsaved_changes = False
    while True:
        print_menu()
        user_input = input("Choose an option: ").strip()
        if user_input == "0":
            if unsaved_changes and ask_yes_no("You have unsaved changes. Save before quitting? (y/n): "):
                save_runs_menu(manager)
            print("Good Bye. Have a nice day")
            break
        elif user_input == "1":
            print_runs(list(manager.get_models().items()))
        elif user_input == "2":
            if add_run(manager):
                unsaved_changes = True
        elif user_input == "3":
            if update_run(manager):
                unsaved_changes = True
        elif user_input == "4":
            if delete_run(manager):
                unsaved_changes = True
        elif user_input == "5":
            sort_runs_menu(manager)
        elif user_input == "6":
            filter_runs_menu(manager)
        elif user_input == "7":
            show_statistics_menu(manager)
        elif user_input == "8":
            show_recommendations_menu(manager)
        elif user_input == "9":
            if save_runs_menu(manager):
                unsaved_changes = False
        elif user_input == "10":
            if load_runs_menu(manager):
                unsaved_changes = True
        else:
            print("Enter a valid choice")


if __name__ == "__main__":
    main()