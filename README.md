🧠 Handwritten Digit Recognition using Deep Learning & Streamlit
📌 Project Overview

This project is a Machine Learning web application that recognizes handwritten digits (0–9) using a Deep Learning model trained on the MNIST dataset.
The trained model is deployed as an interactive Streamlit web app where users can upload a handwritten digit image and get instant predictions.

🎯 Objectives

Understand basics of Deep Learning

Build a Neural Network using TensorFlow & Keras

Train a model on the MNIST dataset

Deploy the trained model using Streamlit

Create a simple and interactive ML web application

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Streamlit

Pillow (PIL)

MNIST Dataset

📂 Project Structure
digit_recognition_app/
│
├── app.py                # Streamlit web application
├── train_model.py        # Model training script
├── mnist_model.h5        # Trained deep learning model
├── requirements.txt      # Required Python libraries
├── README.md             # Project documentation
└── venv/                 # Virtual environment

⚙️ Installation & Setup
Step 1: Clone or Download the Project
git clone <repository-link>


or download the ZIP file and extract it.

Step 2: Create Virtual Environment
python -m venv venv


Activate the environment:

Windows

venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

🧠 Model Training

Run the following command to train the model and generate the .h5 file:

python train_model.py


This will:

Load the MNIST dataset

Train a neural network

Save the trained model as mnist_model.h5

🚀 Running the Streamlit App

Start the web application using:

streamlit run app.py


The app will open automatically in your browser at:

http://localhost:8501

📷 How to Use the Application

Open the Streamlit web app

Upload a handwritten digit image (PNG / JPG)

The image is resized and preprocessed automatically

The model predicts the digit

The predicted digit is displayed on the screen

📊 Dataset Used

MNIST Handwritten Digit Dataset

70,000 grayscale images

Image size: 28×28 pixels

Digits: 0 to 9

🧪 Model Details

Model Type: Artificial Neural Network (ANN)

Input Layer: 28×28 pixel image

Hidden Layer: Dense layer with ReLU activation

Output Layer: Softmax activation (10 classes)

Optimizer: Adam

Loss Function: Categorical Crossentropy

⚠️ Limitations

Model accuracy may decrease for custom images

Sensitive to image alignment and stroke thickness

Performance can be improved using CNN models

🚀 Future Enhancements

Use Convolutional Neural Networks (CNN)

Add drawing canvas for real-time digit input

Improve preprocessing (centering & noise removal)

Deploy application on Heroku / Cloud platform

🎓 Academic Relevance

This project is suitable for:

BCA / MCA Mini Project

Machine Learning coursework

Deep Learning practical implementation

Viva and project demonstrations

👨‍💻 Author

Name: Akash
Course: BCA
Project Type: Mini Project (Machine Learning)

✅ Conclusion

This project successfully demonstrates how a deep learning model can be trained and deployed as a web application using Streamlit. It provides hands-on experience with neural networks, model deployment, and real-world ML challenges.

