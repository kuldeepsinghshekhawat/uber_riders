import numpy as np
import math
from flask import Flask, request, jsonify, render_template
import pickle
app=Flask(__name__)
model=pickle.load(open('taxi.pickle','rb'))
@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(value) for value in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index3.html',prediction_text="Number ofn weekly Rides Should be {}".format(math.floor(output)))
    

if __name__=='__main__':
    app.run(debug=True)


