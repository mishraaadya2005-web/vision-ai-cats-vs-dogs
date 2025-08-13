Vision AI — Cats vs Dogs Classifier 🐱🐶

📌 Overview
This project is part of a 5-day AI bootcamp assignment to build an image recognition system.  
It classifies images of cats and dogs using:
- A Baseline CNN trained from scratch
- Transfer Learning with MobileNetV2 for improved accuracy

All training, evaluation, and inference is done in Google Colab using TensorFlow/Keras.

---

📂 Dataset
- Name: Cats vs Dogs (Microsoft dataset via TensorFlow Datasets)
- Classes: Cat, Dog
- Split:
  - 80% Train
  - 10% Validation
  - 10% Test
- Image Size: 224×224 (resized)

---

🛠️ Approach
1. Data Preprocessing
- Resized all images to 224×224
- Normalized pixel values to [0, 1]
- Applied data augmentation:
  - Random flip
  - Random rotation
  - Random zoom

2. Models
- Baseline CNN  
  Simple 3-layer convolutional neural network trained from scratch
- MobileNetV2 (Transfer Learning)  
  Pretrained on ImageNet, fine-tuned for Cats vs Dogs classification

3. Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

---

## 📊 Results
| Model                   | Accuracy | AUC   |
|-------------------------|----------|-------|
| Baseline CNN            | 86%      | 0.94  |
| MobileNetV2 (Fine-tuned)| 99%      | 0.99  |

---

📈 Visualizations
- Training accuracy/loss curves
- Confusion matrix
- ROC curve
- Sample predictions with probability scores

---

## 🚀 How to Run
1. Clone the repository:
```bash
git clone https://github.com/mishraaadya2005-web/vision-ai-cats-vs-dogs.git


Install dependencies:
pip install -r requirements.txt

Open the notebook in Jupyter or Google Colab:
image_recognition_submittable.ipynb

Demo Video
Watch the 30-second demo
Demo video link - https://drive.google.com/file/d/1OIaRW-IqYy_OAfRVl-3mnLc-fT_doiHW/view?usp=drive_link

📄 Submission Contents
- `image_recognition_submittable.ipynb` — Complete training & evaluation code
- `artifacts/` — Saved models & plots
- `sample_predictions/` — Example outputs
- `requirements.txt` — Dependencies
- `README.md` — Project documentation
- Demo video link - https://drive.google.com/file/d/1OIaRW-IqYy_OAfRVl-3mnLc-fT_doiHW/view?usp=drive_link
- 5-slide presentation (PDF) (Submitted Separately)



Author
Aadya Mishra
BCA Final Year Student 




