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

def main():
    manager = ModelManager()
    while True:
        print_menu()
        user_input = input("Choose an option: ").strip()
        if user_input == "0":
            print("Good Bye. Have a nice day")
            break
        else:
            print("Not implemented yet. Have patience")


if __name__ == "__main__":
    main()