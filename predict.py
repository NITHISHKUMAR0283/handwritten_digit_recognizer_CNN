import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

# 1. Import your custom architecture from your project structure
from src.CNN import CNN

# 2. Setup configuration and device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mnist_cnn.pth"

# 3. Initialize and load your trained model weights
model = CNN()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f" Successfully loaded model weights from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ Warning: Could not find '{MODEL_PATH}'. Please train the model first or check the path.")

model.to(DEVICE)
model.eval()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer (Tkinter Canvas)")
        self.root.resizable(False, False)

        self.image_size = 280
        # Create a black background tracking canvas
        self.pil_image = Image.new("L", (self.image_size, self.image_size), 0)
        self.draw = ImageDraw.Draw(self.pil_image)

        # Left Column: Canvas
        self.canvas = tk.Canvas(root, width=self.image_size, height=self.image_size, bg="black", highlightthickness=1)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Right Column: UI Text Feedback
        self.label_title = tk.Label(root, text="Prediction Results:", font=("Arial", 14, "bold"))
        self.label_title.grid(row=0, column=1, padx=10, pady=5, sticky="n")

        self.label_result = tk.Label(root, text="Draw something...", font=("Arial", 12), justify="left")
        self.label_result.grid(row=0, column=1, padx=10, pady=40, sticky="n")

        self.btn_clear = tk.Button(root, text="Clear Canvas", command=self.clear_all, font=("Arial", 11), bg="#d9534f", fg="white")
        self.btn_clear.grid(row=1, column=1, padx=10, pady=10, sticky="s")

    def paint(self, event):
        # IMPROVEMENT 1: Increased brush radius from 10 to 14. 
        # Tighter lines like "1" are often too faint for a CNN if drawn too thin.
        r = 14  
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

        self.run_inference()

    def clear_all(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.image_size, self.image_size], fill=0)
        self.label_result.config(text="Draw something...")

    def run_inference(self):
        # IMPROVEMENT 2: Auto-cropping bounding boxes.
        # If you draw a tiny '1' in the center, resizing the whole 280x280 canvas 
        # down to 28x28 squashes it completely. We crop tightly to the pixels first.
        bbox = self.pil_image.getbbox()
        
        if bbox:
            # Crop the canvas to just your drawing and pad it slightly to simulate MNIST style
            cropped_image = self.pil_image.crop(bbox)
            processed_image = ImageOps.expand(cropped_image, border=20, fill=0)
        else:
            processed_image = self.pil_image

        # IMPROVEMENT 3: Add Explicit MNIST Normalization Stats
        # Your training script used transforms.ToTensor(), but your inference must match 
        # the global MNIST distribution scaling limits to avoid logit score shifts.
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])

        input_tensor = transform(processed_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_probs, top_classes = torch.topk(probabilities, 3)

        result_text = ""
        for i in range(3):
            digit = top_classes[i].item()
            conf = top_probs[i].item() * 100
            result_text += f"Digit {digit}: {conf:.2f}%\n"
        
        self.label_result.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()