import requests
from flask import Flask, render_template
from flask import request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    model = request.form['model']
    text = request.form['text']
    steps = request.form["steps"]
    seed = request.form['seed']
    print(model,text,steps,seed)
    requests.post('http://ml:8000/stylegan',
                           data={"model": model,
                                 "text": text,
                                 "steps": steps,
                                 "seed": seed})

    return "Model is starting. Check out ./data folder to see it in action."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
