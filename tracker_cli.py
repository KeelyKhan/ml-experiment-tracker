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
        
def add_run(manager):
    model_name = ask_text("Model Name: ")
    training_accuracy = ask_float("Training Accuracy: ")
    training_loss = ask_float("Training Loss: ")
    validation_accuracy = ask_float("Validation Accuracy: ")
    validation_loss = ask_float("Validation Loss: ")
    epochs = ask_int("No of Epochs: ")
    batch_size = ask_int("Batch Size: ")
    learning_rate = ask_float("Learning Rate: ")
    try:
        run = Model(model_name,training_accuracy,training_loss,validation_accuracy,validation_loss,epochs,batch_size,learning_rate)
        run_id = manager.add_model(run)
        print(f"Run added with ID {run_id}")
    except ValueError as e:
        print(f"Could not add this run: {e}")

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
        return
    print("\nChoose a run ID to delete: ")
    print_runs(list(manager.get_models().items()))
    run_id = ask_int("\nEnter run ID: ")
    decision = ask_yes_no(f"Delete run {run_id}? (y/n): ")
    if not decision:
        print("\nCancelled")
        return
    try: 
        manager.delete_model(run_id)
        print(f"Run {run_id} deleted successfully")
    except KeyError:
        print(f"No run with ID {run_id} exists")



def main():
    manager = ModelManager()
    while True:
        print_menu()
        user_input = input("Choose an option: ").strip()
        if user_input == "0":
            print("Good Bye. Have a nice day")
            break
        elif user_input == "1":
            print_runs(list(manager.get_models().items()))
        elif user_input == "2":
            add_run(manager)
        elif user_input == "4":
            delete_run(manager)
        else:
            print("Not implemented yet. Have patience")


if __name__ == "__main__":
    main()