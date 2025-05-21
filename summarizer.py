import os
import nltk

# Force download of proper resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

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

# Calculating the number of exchanges for the summary statistic
def compute_exchanges(user_messages, ai_messages):
    """
    Compute number of exchanges (1 user message + 1 AI response).
    """
    return min(len(user_messages), len(ai_messages))

def compute_message_statistics(user_messages, ai_messages):
    total = len(user_messages) + len(ai_messages)

    return {
        'total': total,
        'user': len(user_messages),
        'ai': len(ai_messages)
    }

#Extracting top keywords by tokenizing the words and removing the stop words("I","you","the" etc)
def extract_top_keywords(user_messages, ai_messages, top_n=5):
    all_text = ' '.join(user_messages + ai_messages)
    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word.isalnum()]


    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)

    return word_counts.most_common(top_n)

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
        
        exchange_count = compute_exchanges(user_msgs, ai_msgs)
        messages_count = compute_message_statistics(user_msgs, ai_msgs)
        top_keywords = extract_top_keywords(user_msgs,ai_msgs)

        print(f"🔁 Number of Exchanges: {exchange_count}")
        print(f"🔁 Number of Messages: {messages_count}")
        for word, count in top_keywords:
            print(f"  - {word}")
            
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

                exchange_count = compute_exchanges(user_msgs, ai_msgs)
                messages_count = compute_message_statistics(user_msgs, ai_msgs)
                top_keywords = extract_top_keywords(user_msgs,ai_msgs)
                print(f"🔁 Number of Exchanges: {exchange_count}")
                print(f"🔁 Number of Messages: {messages_count}")
                print("Top 5 Keywords:")
                for word, count in top_keywords:
                    print(f"  - {word}")
    else:
        print("Please provide a file to start summarizing.")
        
