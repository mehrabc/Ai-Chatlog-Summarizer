import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Part 1: Separating User and AI messages and storing inside distinct arrays
def parse_chat_log(file_path):
    """
    Separating User and AI messages and storing inside distinct arrays for analyzing later.
    """
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
    """
    Generating numerical conversation statistics
    """
    total = len(user_messages) + len(ai_messages)

    return {
        'total': total,
        'user': len(user_messages),
        'ai': len(ai_messages)
    }

#Extracting top keywords by tokenizing the words and removing the stop words("I","you","the" etc)
def extract_top_keywords(user_messages, ai_messages, top_n=5):
    """
    Extracting top keywords
    """
    all_text = ' '.join(user_messages + ai_messages)
    tokens = word_tokenize(all_text)
    tokens = [word for word in tokens if word.isalnum()]


    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)

    return word_counts.most_common(top_n)

#tf-idf approach
def extract_top_keywords_tfidf(user_messages, ai_messages, top_n=5):
    """
    Using tf-idf approach to extract keywords
    """
    # Combine messages into a list of "documents"
    documents = [f"{u} {a}" for u, a in zip(user_messages, ai_messages)]
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

def generate_summary(user_messages, ai_messages):
        exchange_count = compute_exchanges(user_messages, ai_messages)
        messages_count = compute_message_statistics(user_messages, ai_messages)
        top_keywords = extract_top_keywords(user_messages,ai_messages)
        top_keywords_tf_idf = extract_top_keywords_tfidf(user_messages,ai_messages)

        print(f"🔁 Number of Exchanges: {exchange_count}")
        print(f"🔁 Number of Messages: {messages_count}")
        keyword_list = [word for word, _ in top_keywords]
        keyword_list_tfidf = [word for word, _ in top_keywords_tf_idf]

        print("Top 5 keywords (frequency-based): " + ", ".join(keyword_list) + ".")
        print("Top 5 keywords (tf-idf): " + ", ".join(keyword_list_tfidf) + ".")

        tfidf_list = [word for word, _ in top_keywords_tf_idf]
        keywords_for_pos_tagging =" ".join(tfidf_list)
        # Separating the keywords for pos tagging
        tokens = word_tokenize(keywords_for_pos_tagging)
        # Giving the token the parts of speech to identify the nouns
        pos_included = nltk.pos_tag(tokens)
        # Extracting the nouns to include them in the summary
        summary_keyword_extraction = [word for word, tag in pos_included if tag.startswith("NN")]
        selected_keywords = summary_keyword_extraction[:2]

        summary = f"This conversation is about {' and '.join(selected_keywords)}."
        print(summary.capitalize())



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
        generate_summary(user_msgs,ai_msgs)
        



    # Testing if the program can segregate multiple files together
    elif args.folder:
        for filename in os.listdir(args.folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(args.folder, filename)
                print(f"\n📄 File: {filename}")

                user_msgs, ai_msgs = parse_chat_log(filepath)
                generate_summary(user_msgs,ai_msgs)

    else:
        print("Please provide a file to start summarizing.")
        
