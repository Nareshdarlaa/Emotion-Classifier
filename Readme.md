# 🤖 Emotion Classifier using DistilBERT + Gradio

This project demonstrates a real-time **Emotion Detection** system using **DistilBERT**, a lightweight transformer model fine-tuned for emotion classification, with a clean and interactive **Gradio UI**. It classifies text inputs into one of several emotions like **joy**, **sadness**, **anger**, **fear**, **surprise**, or **love**.

---

## 📁 Dataset Used

- **Name:** `dair-ai/emotion`
- **Source:** Hugging Face Datasets ([Link](https://huggingface.co/datasets/dair-ai/emotion))
- **Description:** A curated dataset of text samples annotated with six basic emotions: `joy`, `sadness`, `anger`, `fear`, `surprise`, and `love`.

---

## 🚀 Approach Summary

1. **Text Input** is processed using a pretrained Hugging Face model:
   - `bhadresh-savani/distilbert-base-uncased-emotion`
   - This model captures semantic meaning and context, including **negation handling**.
2. The app is divided into two interactive tabs:
   - **Prediction Tab:** Users input text and receive top emotion predictions with probability scores.
   - **Confusion Matrix Tab:** Displays a confusion matrix on a sample test set to evaluate model performance.
3. **Gradio UI** is used to create a browser-based interface with examples and tabbed layout.

---

## 📦 Dependencies

Install all required packages via pip:

```bash
pip install gradio transformers torch datasets scikit-learn matplotlib
