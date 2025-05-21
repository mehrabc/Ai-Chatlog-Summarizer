import os

# Part 1: Separating User and AI messages and storing inside distinct arrays
def parse_chat_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    user_messages = []
    ai_messages = []

    for line in lines:
        line = line.strip()
        if line.startswith("User:"):
            user_messages.append(line[5:].strip())
        elif line.startswith('AI:'):
            ai_messages.append(line[3:].strip())
    
    return user_messages, ai_messages

# Main function
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ai Chat Log Summarizer')
    parser.add_argument('--file', help='Path to chat log file')
    parser.add_argument('--folder', help='Folder path for batch summarization')

    args = parser.parse_args()
    
    ## Checking if the program are parsing the chatlog correctly.
    if args.file:
        filename = os.path.basename(args.file)
        user_msgs, ai_msgs = parse_chat_log(args.file)
        print(f"\n📄 File: {filename}")
        print("👤 User Messages:")
        for msg in user_msgs:
            print(f"  - {msg}")

        print("🤖 AI Messages:")
        for msg in ai_msgs:
            print(f"  - {msg}")
    # Testing if the program can segregate multiple files together
    elif args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(args.folder, filename)
                print(f"\n📄 File: {filename}")

                user_msgs, ai_msgs = parse_chat_log(filepath)
                print("👤 User Messages:")
                for msg in user_msgs:
                    print(f"  - {msg}")

                print("🤖 AI Messages:")
                for msg in ai_msgs:
                    print(f"  - {msg}")


    else:
        print("Please provide a file to start summarizing.")
        
