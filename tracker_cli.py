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
        print("No runs to show")
        return 
    print(f"{'ID':<4}{'Name':<12}{'Train Acc':<11}{'Val Acc':<11}{'Train Loss':<11}{'Val Loss':<11}{'Epochs':<8}{'Batch':<8}{'LR':<10}")
    for run_id,run in pairs:
        print(f"{run_id:<4}{run.model_name:<14}{run.training_accuracy:<11.2f}{run.validation_accuracy:<11.2f}{run.training_loss:<11.2f}{run.validation_loss:<11.2f}{run.epochs:<8}{run.batch_size:<8}{run.learning_rate:<10g}")


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
        else:
            print("Not implemented yet. Have patience")


if __name__ == "__main__":
    main()