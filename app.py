import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
model = pickle.load(open(r'C:\Users\OLUWASEUN ADEGOROYE\Downloads\heart-detector-master\heart-detector-master\Heart_disease_detector\Mikail_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['age', 'sex', 'restingbp', 'chestpaintype', 'cholesterol', 'fastingbs', 'restingecg',
                     'maxhr', 'exerciseangina', 'oldpeak', 'st_slope']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "** Heart Disease **"
    else:
        res_val = "No Heart Disease "

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))


if __name__ == "__main__":
    app.debug = True
    app.run()
