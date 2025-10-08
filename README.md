# 📰 Fake News Detector – Streamlit App

A clean and interactive web app to classify news content as **Real** or **Fake** using a fine-tuned **BERT model**.  
Built with **Streamlit**, this app provides an elegant interface with instant predictions and confidence visualization.

[➡️ Try the Live App on Streamlit](https://bert-tasks---fake-news-detection-dataset-wurqcmbg5ojfq8xj8audb.streamlit.app/)

---

## 🖼️ Screenshots

### Home Page
![Home Page](screenshots/home_page.png)

### Prediction Result
![Prediction Result](screenshots/prediction_result.png)

---

## 🚀 Features

- 🧠 Uses BERT (`bert-base-uncased`) fine-tuned for fake news detection  
- ⚡ Fast and responsive **Streamlit UI**  
- 📈 Displays **confidence scores** for Fake and Real classes  
- 🎨 Minimal design with purple–indigo theme  
- 🌍 Fully deployable on **Hugging Face Spaces** or **Streamlit Cloud**

---

## 🧪 Tech Stack

- [Transformers](https://huggingface.co/transformers/) – for loading the BERT model  
- [PyTorch](https://pytorch.org/) – for model inference  
- [Streamlit](https://streamlit.io/) – for the interactive web interface  
- [Hugging Face Hub](https://huggingface.co/docs/hub/) – for model hosting and versioning

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/subhiksha-kodi/BERT-TASKS---FAKE-NEWS-DETECTION-DATASET.git
cd BERT-TASKS---FAKE-NEWS-DETECTION-DATASET
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```text
streamlit
torch
transformers
huggingface-hub
numpy
pandas
```

### 3️⃣ Ensure your model is uploaded to Hugging Face Hub

Example model repo: `subhiksha-kodi/bert-fake-news-model`

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

This will start a local Streamlit server and display a URL like:

```
Local URL: http://localhost:8501
```

Open it in your browser to test the Fake News Detector.

---

## 📚 Model Info

- **Base Model:** bert-base-uncased  
- **Fine-tuned on:** Fake News Detection Dataset (balanced real/fake samples)  
- **Framework:** Hugging Face Transformers + PyTorch  
- **Accuracy:** ~100% on validation set (after 1 epoch)  

---

## 🌐 Deployment Options

You can deploy this Streamlit app easily on:

- ☁️ **Streamlit Cloud** – simple and free hosting  
- ☁️ **Hugging Face Spaces** – using Streamlit SDK  
- 💻 **Local Machine** – run with `streamlit run app.py`  

---

## 👨‍💻 Author

**Subhiksha Kodibass** – [GitHub](https://github.com/subhiksha-kodi)

---

## 📄 License

This project is licensed under the **MIT License**.

