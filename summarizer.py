import os
import nltk

# Force download of proper resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

#tf-idf approach
def extract_top_keywords_tfidf(user_messages, ai_messages, top_n=5):
    # Combine messages into a list of "documents"
    documents = user_messages + ai_messages
    # Create TF-IDF Vectorizer (with English stopwords removed)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
     # Sum TF-IDF scores for each term across all documents
    summed_tfidf = tfidf_matrix.sum(axis=0)
    scores = np.asarray(summed_tfidf).flatten()
    # Map terms to scores
    term_scores = list(zip(vectorizer.get_feature_names_out(), scores))

    # Sort by score and get top N
    top_keywords = sorted(term_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return top_keywords

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
        top_keywords_tf_idf = extract_top_keywords_tfidf(user_msgs,ai_msgs)

        print(f"🔁 Number of Exchanges: {exchange_count}")
        print(f"🔁 Number of Messages: {messages_count}")
        keyword_list = [word for word, _ in top_keywords]
        print("Top 5 keywords (frequency-based): " + ", ".join(keyword_list) + ".")

        tfidf_list = [word for word, _ in top_keywords_tf_idf]
        print("Top 5 keywords using TF-IDF are: " + ", ".join(tfidf_list) + ".")

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
                top_keywords_tf_idf = extract_top_keywords_tfidf(user_msgs,ai_msgs)

                print(f"🔁 Number of Exchanges: {exchange_count}")
                print(f"🔁 Number of Messages: {messages_count}")

                keyword_list = [word for word, _ in top_keywords]
                print("Top 5 keywords (frequency-based): " + ", ".join(keyword_list) + ".")

                tfidf_list = [word for word, _ in top_keywords_tf_idf]
                print("Top 5 keywords using TF-IDF are: " + ", ".join(tfidf_list) + ".")
    else:
        print("Please provide a file to start summarizing.")
        
