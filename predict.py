import tkinter as tk
from tkinter import ttk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tkinter.font import Font

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./spam_classifier_model')
tokenizer = BertTokenizer.from_pretrained('./spam_classifier_model')

# Define prediction function
def predict(message):
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Spam" if prediction == 1 else "Ham"

# Function to classify a single message
def classify_message():
    message = input_text.get("1.0", tk.END).strip()
    if not message:
        result_label.config(text="Please enter a message!")  # Orange for warnings
        return
    result_label.config(text="Processing...", fg="#0000ff")  # Blue for processing
    root.update_idletasks()  # Update the window to show the processing status
    result = predict(message)
    result_label.config(
        text=f"Prediction: {result}",
        fg="#FF0000" if result == "Spam" else "#008000",
    )
# Function to clear input and result fields
def clear_fields():
    input_text.delete("1.0", tk.END)
    result_label.config(text="")

# Create the Tkinter GUI
root = tk.Tk()
root.title("Spam Message Classifier")
root.configure(bg="#F0FFF0")  # Light green background

# Center the window
window_width = 750
window_height = 550
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Custom Fonts
title_font = Font(family="Helvetica", size=22, weight="bold")
label_font = Font(family="Arial", size=12)
button_font = Font(family="Arial", size=12, weight="bold")
result_font = Font(family="Arial", size=16, weight="bold")

# Create a main frame for centering content
main_frame = tk.Frame(root, bg="#228B22", height=70)
main_frame.pack(fill="x")

# Header
header_label = tk.Label(
    main_frame,
    text="Spam Classifier",
    font=title_font,
    bg="#228B22",
    fg="white",
)
header_label.pack(pady=15)

# Input Frame
input_frame = tk.Frame(root, bg="#F0FFF0", pady=10)
input_frame.pack(fill="x", padx=20)
# Input Area
input_label = tk.Label(
    input_frame, 
    text="Enter your message:", 
    font=label_font, 
    bg="#F0FFF0"
)
input_label.pack(anchor="w", pady=(10, 5))

input_text = tk.Text(
    input_frame, 
    height=8,
    width=65,
    font=("Arial", 12),
    bd=0,
    relief="solid",
    padx=10,
    pady=10,
    bg="#FFFFFF",
    highlightbackground="#B0E57C",  # Soft green border
    highlightthickness=1,
)
input_text.pack(pady=(0, 20))

# Create a frame for buttons with grid layout
button_frame = tk.Frame(root, bg="#F0FFF0")
button_frame.pack(fill="x", pady=(5, 10))


classify_button = tk.Button(
    button_frame, 
    text="Predict", 
    command=classify_message, 
    font=button_font,
    bg="#32CD32",  # Lime green
    fg="white",
    bd=0,
    padx=20,
    pady=10,
    activebackground="#2E8B57",  # Sea green
    activeforeground="white",
    relief="raised",
    highlightthickness=0,
    cursor="hand2",
)
classify_button.pack(side="left", padx=(120, 10))

clear_button = tk.Button(
    button_frame, 
    text="Clear", 
    command=clear_fields, 
    font=button_font,
    bg="#32CD32",  # Lime green
    fg="white",
    bd=0,
    padx=20,
    pady=10,
    activebackground="#2E8B57",
    activeforeground="white",
    relief="raised",
    highlightthickness=0,
    cursor="hand2",
)
clear_button.pack(side="left", padx=(10, 120))

# Result Frame
result_frame = tk.Frame(root, bg="#ffffff", pady=10)
result_frame.pack(fill="x", expand=True)

result_label_frame = tk.LabelFrame(result_frame, text="Prediction Result", font=("Arial", 12, "bold"), bg="#ffffff", fg="#333")
result_label_frame.pack(fill="x", padx=20, pady=10)

result_label = tk.Label(result_label_frame, text="", font=("Arial", 14), bg="#ffffff", fg="#333")
result_label.pack(pady=10, padx=10)

# Footer
footer_label = tk.Label(
    root,
    text="Developed by Shiksha Jaiswal © 2024",
    font=("Arial", 10),
    bg="#F0FFF0",
    fg="#777",
)
footer_label.pack(side="bottom", pady=10)

# Run the Tkinter event loop
root.mainloop()
