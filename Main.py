import gradio as gr
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset for CM
dataset = load_dataset("dair-ai/emotion")
df = dataset["train"].to_pandas()
df["label"] = df["label"].map(lambda i: dataset['train'].features['label'].names[i])
df_test = dataset["test"].to_pandas()
df_test["label"] = df_test["label"].map(lambda i: dataset['test'].features['label'].names[i])

# Load pretrained emotion classification model from Hugging Face
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Prediction function
def predict_emotion(text):
    result = classifier(text)[0]
    return {entry['label']: round(entry['score'], 3) for entry in sorted(result, key=lambda x: x['score'], reverse=True)}

# Confusion matrix plot
def plot_confusion_matrix():
    try:
        sample_df = df_test.sample(300, random_state=42)  # sample smaller set to avoid timeout
        y_true = sample_df["label"]
        y_pred = []

        for text in sample_df["text"]:
            try:
                preds = classifier(text)
                top_label = max(preds[0], key=lambda x: x['score'])['label']
                y_pred.append(top_label)
            except Exception as e:
                y_pred.append("unknown")

        # Filter out any bad predictions
        filtered = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp != "unknown"]
        y_true_filtered = [yt for yt, yp in filtered]
        y_pred_filtered = [yp for yt, yp in filtered]

        labels = sorted(set(y_true_filtered + y_pred_filtered))
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Oranges", xticks_rotation=45)
        plt.title("Confusion Matrix (Sample of 300)")
        return fig
    except Exception as e:
        print("Error generating confusion matrix:", e)
        raise gr.Error("Failed to generate confusion matrix. Try again.")


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Emotion Classifier using DistilBERT\nType a sentence to detect its emotional tone.")
    
    with gr.Tab("💬 Predict Emotion"):
        inp = gr.Textbox(placeholder="E.g. I don't want to talk to you...", lines=3, label="Enter your sentence")
        out = gr.Label(num_top_classes=4)
        btn = gr.Button("🔍 Predict")
        btn.click(fn=predict_emotion, inputs=inp, outputs=out)
    
    with gr.Tab("📊 Confusion Matrix"):
        cm_btn = gr.Button("Generate Confusion Matrix")
        cm_plot = gr.Plot()
        cm_btn.click(fn=plot_confusion_matrix, outputs=cm_plot)

    gr.Examples(["I am very happy today!", 
                 "I don't want to talk to you anymore.", 
                 "I'm scared of the result.", 
                 "He makes me angry."], 
                inputs=inp)

demo.launch()
