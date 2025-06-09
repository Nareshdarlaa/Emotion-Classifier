# Emotion Classifier 

This project is an interactive emotion classifier that takes a sentence as input and predicts the most likely emotional tone—such as Happy, Sad, Angry, or Neutral. It uses the TF-IDF text vectorizer and the powerful XGBoost classifier for robust and accurate predictions. The application is deployed with a user-friendly interface using Gradio.

---

## Dataset Used

**[DAIR-AI Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)**  
- Source: Hugging Face `datasets` library  
- Type: Text classification  
- Classes: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`  
- Format: Each sample consists of a sentence and an associated emotion label.

---

## Approach Summary

1. **Data Loading**  
   - Loaded from the Hugging Face `dair-ai/emotion` dataset.
   - Converted to Pandas DataFrame for manipulation.

2. **Preprocessing**  
   - Removed English stopwords.
   - Converted sentences into TF-IDF vectors with a vocabulary size limit (`max_features=5000`).

3. **Modeling**  
   - Used `XGBoost Classifier` for multi-class classification.
   - Evaluated with a **confusion matrix** on the test data.

4. **UI with Gradio**  
   - Built an interactive UI with tabs:
     - **Predict Emotion**: Type text and get predicted emotions with probabilities.
     - **Confusion Matrix**: Visualizes model performance on test data.
   - Provided examples for quick testing.

---

## How to Run

### Clone the Repo
```bash
git clone https://github.com/your-username/emotion-classifier-xgboost.git
cd emotion-classifier-xgboost
