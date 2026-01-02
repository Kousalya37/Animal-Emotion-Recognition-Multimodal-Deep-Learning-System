# 🐾 Explainable Multimodal Deep Learning System for Animal Emotion Recognition

> A cutting-edge AI framework that combines audio, visual, and biometric data to recognize and **explain animal emotions in real-time**. Built with transformer-based models, multimodal fusion, and explainable AI (XAI) powered by GPT-4.

---

## Overview
Animal emotion recognition is a growing field with applications in **veterinary medicine, animal welfare, and human-animal interaction**.  
Traditional unimodal approaches (only audio or only images) often fail in noisy, dynamic environments.  
This project introduces a **multimodal deep learning system** that integrates:

- **Audio** (animal vocalizations) → Wav2Vec2 Transformer  
- **Images** (faces & body cues) → Vision Transformer (OpenCLIP ViT-B/32)  
- **Biometrics** (heart rate, temperature, variability) → Normalized numerical features  

All modalities are fused using **LSTM + Dense layers**, producing robust emotion classification across **7 categories**:  
`Happy, Sad, Angry, Fearful, Relaxed, Curious, Neutral`

To ensure transparency, predictions are explained in **natural language** using GPT-4.

---

##  Key Features
- **Multimodal Fusion**: Audio + Image + Biometric integration  
- **Transformer-based Feature Extraction**: Wav2Vec2 & Vision Transformer  
- **Interactive Visualization**: Real-time emotion distribution via Chart.js pie charts  
- **Explainable AI (XAI)**: GPT-4 generates human-readable explanations for predictions  
- **Real-time Inference**: Lightweight Flask REST API backend   
- **Species Adaptability**: Extendable to new animals, sensors, or contexts  

<img width="940" height="1200" alt="image" src="https://github.com/user-attachments/assets/be1ac63b-f92a-4b23-a634-6f0de6558c01" />
