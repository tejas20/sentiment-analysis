import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))
tv_lr_model = pickle.load(open('LR_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input = tv_lr_model.transform([request.form['ReviewText']]).toarray() 
    prediction = model.predict(input)
    confidence_score = model.predict_proba(input)
    
    if prediction[0] == 1:
        output = "Positive"
    else:
        output = "Negative"
        
    return render_template('index.html', prediction_text=output,
                          confidence_score_text=confidence_score)

    
if __name__ == "__main__":
    app.run(debug=True)
    app.config["DEBUG"]=True
