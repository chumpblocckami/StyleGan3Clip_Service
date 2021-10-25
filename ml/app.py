from flask import Flask
from flask import request

from StyleGan import StyleGan as SGS

app = Flask(__name__)
service = SGS()


@app.route('/stylegan', methods=['POST'])
def run_stylegan():
    if request.method == "POST":
        try:
            model = request.form.get("model")
            text = request.form.get("text")
            seed = int(request.form.get("seed"))
            steps = int(request.form.get("steps"))
        except Exception as typeError:
            return str(typeError)
        print(f'Apply GAN using: {model},{text},{seed},{steps}')
        service.run_update(model=model,
                        text=text,
                        seed=seed,
                        steps=steps)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)