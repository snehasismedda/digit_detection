import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib

# Load the scaler and models
scaler = joblib.load('scaler.pkl')
models = {
    'Logistic Regression': joblib.load('Logistic_Regression.pkl'),
    'Random Forest': joblib.load('Random_Forest.pkl'),
    'Decision Tree': joblib.load('Decision_Tree.pkl'),
    'SVM': joblib.load('SVM.pkl'),
    'KNN': joblib.load('KNN.pkl'),
    'Neural Network': joblib.load('Neural_Network.pkl')
}

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
        
        # Majority voting
        final_prediction = max(predictions.values(), key=list(predictions.values()).count)
        
        result_text = "\n".join([f"{name}: {pred}" for name, pred in predictions.items()])
        result_text += f"\n\nFinal Prediction: {final_prediction}"
        
        self.label_result.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
