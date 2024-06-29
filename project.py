import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load the digits dataset
digits = load_digits()
X = digits.images
y = digits.target

# Standardize the dataset
scaler = StandardScaler()
X = X.reshape((X.shape[0], -1))
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape data for CNN
X_train_cnn = X_train.reshape((X_train.shape[0], 8, 8, 1))
X_test_cnn = X_test.reshape((X_test.shape[0], 8, 8, 1))

# Dictionary to store the models and their names
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

# Add a simple CNN model
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, to_categorical(y_train), epochs=20, batch_size=32, validation_split=0.2, verbose=0)

# Train and evaluate each model
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy}")
    print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))

# Evaluate CNN model
cnn_predictions = cnn_model.predict(X_test_cnn)
cnn_pred_labels = np.argmax(cnn_predictions, axis=1)
cnn_accuracy = accuracy_score(y_test, cnn_pred_labels)
accuracies['CNN'] = cnn_accuracy
print(f"CNN Accuracy: {cnn_accuracy}")
print(f"Classification Report for CNN:\n", classification_report(y_test, cnn_pred_labels))

# Compare the accuracies
plt.figure(figsize=(12, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta'])
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Methods')
plt.ylim(0, 1)
plt.show()

# Tkinter GUI for Digit Recognizer with Majority Voting
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")
        
        self.canvas = tk.Canvas(master, width=200, height=200, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky="W")
        
        self.button_predict = tk.Button(master, text="Predict", command=self.predict_digit)
        self.button_predict.grid(row=1, column=0, pady=2, padx=2)
        
        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1, pady=2, padx=2)
        
        self.label_result = tk.Label(master, text="Draw a digit and click 'Predict'")
        self.label_result.grid(row=2, column=0, pady=2, padx=2)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit and click 'Predict'")

    def predict_digit(self):
        img = self.image.resize((8, 8), Image.LANCZOS)
        img = ImageOps.invert(img)
        img = np.array(img).reshape(1, -1)
        img = scaler.transform(img)
        
        # Get predictions from all models
        predictions = {name: model.predict(img)[0] for name, model in models.items()}
        
        # Get prediction from CNN model
        img_cnn = img.reshape((1, 8, 8, 1))
        cnn_prediction = np.argmax(cnn_model.predict(img_cnn), axis=1)[0]
        predictions['CNN'] = cnn_prediction
        
        # Majority voting
        final_prediction = max(predictions.values(), key=list(predictions.values()).count)
        
        result_text = "\n".join([f"{name}: {pred}" for name, pred in predictions.items()])
        result_text += f"\n\nFinal Prediction: {final_prediction}"
        
        self.label_result.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
