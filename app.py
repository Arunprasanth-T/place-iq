from flask import Flask, render_template, request
import numpy as np
import pickle

# Load your model (ensure placement.pkl is in the same folder as this script)
model = pickle.load(open('placement.pkl', 'rb'))

app = Flask(__name__)

# CHANGE 1: The home route must load the form (front.html)
@app.route('/')
def home():
    return render_template('front.html')

# CHANGE 2: The predict route must handle the POST data
@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form names in front.html
    data1 = request.form['id']
    data2 = request.form['gender']
    data3 = request.form['marks1']
    data4 = request.form['boards1']
    data5 = request.form['marks2']
    data6 = request.form['boards2']
    data7 = request.form['strm']
    data8 = request.form['deg_p']
    data9 = request.form['deg_s']
    data10 = request.form['wrx']
    data11 = request.form['amcat']
    data12 = request.form['sp']
    data13 = request.form['mba_p']
    data14 = request.form['sal']

    # CHANGE 3: Numerical conversion is handled here by dtype=float
    arr = np.asarray([[data1, data2, data3, data4, data5, data6, data7, data8,
                       data9, data10, data11, data12, data13, data14]], dtype=float)
    
    # Run the prediction
    pred = model.predict(arr)
    
    # CHANGE 4: Return 'after.html' (your result template) with pred[0]
    return render_template('after.html', data=pred[0], 
                           data1=data1, data2=data2, data3=data3,
                           data4=data4, data5=data5, data6=data6, 
                           data7=data7, data8=data8, data9=data9, 
                           data10=data10, data11=data11, data12=data12,
                           data13=data13, data14=data14)

if __name__ == "__main__":
    app.run(debug=True)