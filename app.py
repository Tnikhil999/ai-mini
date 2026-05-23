from flask import Flask, render_template, request
import pickle
import numpy as np
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("open_key")
)

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    study = float(request.form['study'])
    sleep = float(request.form['sleep'])
    previous = float(request.form['previous'])

    if study + sleep > 24:
        return render_template(
            'index.html',
            prediction_text='Error: Study hours + Sleep hours cannot exceed 24'
        )

    features = np.array([[study, sleep, previous]])

    prediction = model.predict(features)

    output = round(prediction[0], 2)

    prompt = f"""
    A student studies {study} hours,
    sleeps {sleep} hours,
    and previously scored {previous}.

    Predicted score is {output}.

    Give short practical advice in 2-3 sentences.
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    advice = response.choices[0].message.content

    try:
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=20
    )

    advice = response.choices[0].message.content

except Exception as e:
    advice = "AI advice temporarily unavailable."

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
