from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['news']
        transformed_input = vectorizer.transform([input_text])
        prediction = model.predict(transformed_input)[0]
        label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
        return render_template('index.html', prediction_text=label, news_input=input_text)

if __name__ == '__main__':
    app.run(debug=True)