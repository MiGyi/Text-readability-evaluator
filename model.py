#word segmenter
import py_vncorenlp
import os
    
#py_vncorenlp.download_model(save_dir='/VnCoreNLP')

here = os.path.dirname(os.path.abspath('model.py'))
os.chdir(here)

rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=os.path.join(here, 'VnCoreNLP'))

import numpy
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModel
from keras.preprocessing.sequence import pad_sequences
import torch

# Load pre-trained PhoBERT tokenizer and model
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")

# Define a function to tokenize and extract features from text
def extract_features(text):
    #process text
    text = rdrsegmenter.word_segment(text)
    text = " ".join(text)
    ids = phobert_tokenizer.encode(text)
    ids_padded = pad_sequences([ids], maxlen=256, dtype="long", value=0, truncating="post", padding="post")
    ids_padded = ids_padded[0]
    mask = [int(token_id > 0) for token_id in ids_padded]
    ids_input = torch.tensor(ids_padded).to(torch.long).reshape(1,-1)
    input_mask = torch.tensor(mask).reshape(1,-1)
    
    with torch.no_grad():
        features = phobert_model(input_ids=ids_input, attention_mask=input_mask)
    
    return features[0][:, 0, :].numpy()
   
def predict(text, model):
    features = extract_features(text).reshape(1,-1)
    prediction = model.predict(features)
    return prediction    
    
#main    
import pickle

#load
with open(here + '/model.pkl', 'rb') as f:
    svc_model = pickle.load(f)
    
#predict
#text = "Tôi là sinh viên trường đại học bách khoa hà nội"
#prediction = predict(text, svc_model)
#print(prediction)

import tkinter as tk

def evaluate_button_click():
    input_data = text_box.get("1.0", "end-1c")
    result = predict(input_data, svc_model)
    if result == 'veryeasy':
        result = 'very easy'
    if result == 'difficult':
        result = 'difficult'
    if result == 'easy':
        result = 'easy'
    if result == 'medium':
        result = 'medium'
    result = str(result)
    result_label.config(text="The given text is " + result)
    
# Create the GUI
window = tk.Tk()
window.title("Text Difficulty Evaluator")

text_box = tk.Text(window, height=5, width=50)
text_box.pack(pady=10)

evaluate_button = tk.Button(window, text="Evaluate", command=evaluate_button_click)
evaluate_button.pack()

result_label = tk.Label(window, text="")
result_label.pack(pady=10)

window.mainloop()