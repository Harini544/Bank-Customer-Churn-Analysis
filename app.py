
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template,redirect,url_for,session,flash
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your dataset
# Assuming you have a CSV file named 'your_dataset.csv'
# df = pd.read_excel('Churn_data.xlsx')

# # Drop the target variable for training
# X = df.drop('Exited', axis=1)

# # Convert categorical variables to numerical
# label_encoder = LabelEncoder()
# X['Gender'] = label_encoder.fit_transform(X['Gender'])

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, df['Exited'], test_size=0.2, random_state=42)

# # Train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

#model = pickle.load(open('model.pkl','rb'))
model = pickle.load(open('model.pklz', 'rb'))


#load model
def load_model():
    global model
    model=pickle.load(open("model.pklz","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_model()
    features = [request.form[x] for x in request.form.values()]
    #features[1] = label_encoder.transform([features[1]])[0]  # Convert Gender to numerical
    #prediction = model.predict([features])[0]

    prediction = model.predict(features)
    if prediction[0]==0:
        a="Not Churned"
    elif prediction[0]==1:
        a="Churned"

    return render_template("index.html", prediction_text="Bank Customer is : {}".format(a))
    #return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,port=5500)
