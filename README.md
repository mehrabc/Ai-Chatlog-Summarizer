# 🤖 AI Chat Log Summarizer App

This is a Python-based command-line tool that reads AI chat logs (in `.txt` format), extracts key information such as message statistics and keywords, and generates a short summary of each conversation using NLP techniques.

---

## 🧠 Features

- Extracts and separates **user vs AI messages**
- Computes:
  - Total messages
  - Number of exchanges
- Extracts **top keywords** using:
  - Frequency-based method
  - TF-IDF (Term Frequency–Inverse Document Frequency)
- POS tagging to generate concise one-line summaries based on top **nouns**
- (BONUS): Batch processing for entire folders of chat logs
- 
## 🧾 Sample Output:
![image](https://github.com/user-attachments/assets/39e843be-385a-428d-b409-ed1b34e472ed)

## 🧩 Installation & Setup

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/chat-log-summarizer.git
cd chat-log-summarizer
pip install nltk scikit-learn numpy

🚀 Running the App
✅ Summarize a Single File:
python summarizer.py --file conversations/chat4_job_search.txt
✅ Summarize All Files in a Folder:
python summarizer.py --folder conversations/



