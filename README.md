# **Email Spam/Ham Classifier**

---

### **Overview**
This project presents a straightforward yet powerful solution for classifying email messages as either **"Spam" (unwanted)** or **"Ham" (legitimate)**. It leverages **transfer learning** with a **Hugging Face Transformers** model, accessible via an API hosted on Hugging Face Spaces, and an intuitive web interface for easy interaction.

### **Features**
* **Spam/Ham Classification**: Utilizes an advanced machine learning model for accurate email categorization.
* **API-driven**: The core classification logic is exposed via a robust API hosted on **Hugging Face Spaces**, allowing for seamless integration into other applications.
* **Simple Web Interface**: A user-friendly web-based UI provides a convenient way to test the model directly in your browser.

---

### **Underlying Model**
The core of this classifier is a fine-tuned **`distilbert-base-uncased`** model.

* **Model Type**: This is an **encoder-only Transformer model**, optimized for understanding text and performing classification tasks. It leverages the vast linguistic knowledge gained during its pre-training on a massive English text corpus (like Wikipedia and BooksCorpus).
* **Fine-tuning**: The `distilbert-base-uncased` model was specifically **fine-tuned** on a custom dataset of email messages to learn the distinct patterns of spam and ham. This specialized training allows it to achieve high accuracy for email classification.
* **Model Size**: The trained model weighs approximately **268 MB**, making it efficient for deployment.

---

### **Project Structure**


```
├── app.py                      # (Or your main Flask/FastAPI backend API script)
├── model/                      # Directory to store the fine-tuned model and tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
├── data/                       # Directory containing the dataset
│   └── spam_data.csv           # Your dataset file (example name)
├── static/                     # (If your HTML/CSS/JS are in a static folder for your web app)
│   ├── index.html              # Simple web interface for testing
│  
│   
├── requirements.txt            # List of Python dependencies
├── .flaskenv                   # (If you're using Flask, e.g., for local development)
├── spam_classifier_finetuning.py # (Your training script, if kept separate from app.py)
└── README.md                   # This documentation file

  ``` 


*Note: Adjust the structure above to accurately reflect your project's actual file organization.*

---

### **Technical Stack**
* **Python**: The primary programming language used.
* **Hugging Face Transformers**: For pre-trained models and efficient fine-tuning.
* **PyTorch**: The deep learning framework used by the Transformer model.
* **Pandas**: For data manipulation and reading the dataset.
* **scikit-learn**: For data splitting (train/test).
* **Hugging Face Datasets**: For efficient dataset loading and preprocessing.
* **Hugging Face Evaluate**: For robust model evaluation metrics (Accuracy, F1-Score).
* **HTML/CSS/JavaScript**: For the web interface.

---

### **Dataset**
The model was fine-tuned on a custom dataset of email messages, labeled as either `ham` (legitimate) or `spam` (unwanted).

* **Format**: Typically a CSV file with columns like `text` (for email content) and `label` (`ham` or `spam`).
* **Note**: For privacy reasons, the original dataset is not included in this repository. The fine-tuning process was conducted using an external dataset.

---

### **Installation**
To set up and run this project locally, follow these steps:

#### **1. Environment Setup**
Ensure you have Python 3.9+ installed. It's highly recommended to use a virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

python app.py
```


## How to Run (Live Demo)

You can test the Email Spam/Ham Classifier directly via its hosted web interface:

🔗 **Live Web Interface**: [https://abdhilal.github.io/emails-spam-ham/](https://abdhilal.github.io/emails-spam-ham/)  
🔗 **Hugging Face Spaces API**: The backend classification model is hosted on Hugging Face Spaces.

---

##  Contact

For any questions, feedback, or collaborations, feel free to reach out:

- 👤 **Your Name**: Abdalrhman Hilal  
- 🔗 [LinkedIn](https://www.linkedin.com/in/abdalrhman-hilal/)  
- 🌐 [Portfolio](https://abdr-hilal.ct.ws/)
