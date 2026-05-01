# 🤖 TinyGPT

A lightweight GPT-style language model built from scratch using PyTorch.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

---

## 🧠 About

TinyGPT is a minimal implementation of a **Transformer-based language model** inspired by GPT architecture.

This project is built for:
- 📚 Learning how LLMs work
- 🧪 Experimentation
- 🛠️ Building custom AI models from scratch

---

## ⚡ Features

- 🔤 Token embedding
- 🧠 Self-attention mechanism
- 🧩 Transformer decoder architecture
- 📉 Training loop with loss optimization
- ✍️ Text generation from prompts
- 🧱 Clean and beginner-friendly code

---

## 🗂️ Project Structure


tinygpt/
│── model.py # Model architecture
│── train.py # Training loop
│── generate.py # Text generation
│── tokenizer.py # Tokenizer
│── dataset.txt # Training data
│── utils.py # Helper functions
│── config.py # Configurations
│── model.pt # Trained model


---

## 🛠️ Installation

```bash
git clone https://github.com/Abeesh-2027/tinygpt.git
cd tinygpt
pip install -r requirements.txt

Usage
Train the model
python train.py

Generate text
python generate.py

Example
Input
  Once upon a time

Output
   Once upon a time there was a small AI learning to write its own story...

🎯 Future Improvements
🔥 Add BPE tokenizer
🌐 Build web interface (Streamlit)
💬 Convert into chatbot
⚡ Optimize training speed
📊 Add attention visualization
🤝 Contributing


