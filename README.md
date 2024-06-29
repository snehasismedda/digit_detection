# Digit Recognizer with Machine Learning Models

This project implements a digit recognition application using various machine learning models. The application includes a Tkinter-based GUI where users can draw digits and get predictions using multiple classifiers.

## Features

- **Tkinter GUI**: Allows users to draw digits using a canvas and predict the digit using machine learning models.
- **Machine Learning Models**: Includes several models trained on the MNIST-like digits dataset.
- **Majority Voting**: Combines predictions from multiple models using majority voting for enhanced accuracy.


## Usage

1. **Run the GUI Application:**
    ```bash
    python main.py
    ```
    - This command will start the Tkinter GUI where you can draw digits and get predictions.

## Models Included

- **Logistic Regression**
- **Random Forest**
- **Decision Tree**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Multilayer Perceptron (Neural Network)**

## File Structure

```bash
digit-recognizer/
│
├── main.py           # Tkinter GUI implementation
├── model.py          # Model training and saving
├── requirements.txt  # Python dependencies
├── scaler.pkl        # Saved scaler for data preprocessing
├── Logistic_Regression.pkl  # Trained Logistic Regression model
├── Random_Forest.pkl         # Trained Random Forest model
├── Decision_Tree.pkl         # Trained Decision Tree model
├── SVM.pkl                   # Trained SVM model
├── KNN.pkl                   # Trained KNN model
├── Neural_Network.pkl        # Trained Neural Network model
└── README.md         # Project documentation
