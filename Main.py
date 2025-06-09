pip install datasets
rm -rf ~/.cache/huggingface/datasets
pip install -U datasets fsspec
!pip install xgboost
import gradio as gr
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from xgboost import XGBClassifier

# Load dataset
dataset = load_dataset("dair-ai/emotion")
df = dataset['train'].to_pandas()

# Preprocess
X = df['text']
y = df['label']  # numeric labels already

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Function to plot confusion matrix as PIL image
def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred)
    labels = dataset['train'].features['label'].names
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img

# Prediction function for Gradio (outputs label probabilities)
def predict_emotion(text):
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    labels = dataset['train'].features['label'].names
    top_classes = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)
    return {label: round(score, 3) for label, score in top_classes}

# Gradio interface with tabs
with gr.Blocks() as iface:
    gr.Markdown("# Emotion Classifier (XGBoost + TF-IDF)")
    with gr.Tabs():
        with gr.TabItem("Predict Emotion"):
            input_text = gr.Textbox(lines=3, placeholder="Type a sentence to detect emotion...")
            output_label = gr.Label(num_top_classes=4)
            btn_predict = gr.Button("Predict")
            btn_predict.click(predict_emotion, inputs=input_text, outputs=output_label)

        with gr.TabItem("Confusion Matrix"):
            cm_image = gr.Image()
            btn_cm = gr.Button("Show Confusion Matrix")
            btn_cm.click(plot_confusion_matrix, inputs=None, outputs=cm_image)

iface.launch()
