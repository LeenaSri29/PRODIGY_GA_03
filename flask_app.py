"""
flask_app.py
Flask web frontend for Markov text generator
"""

from flask import Flask, request, render_template_string
from markov_advanced import MarkovChain

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
<title>Markov Text Generator</title>
<style>
body { font-family: sans-serif; margin: 2em; }
textarea { width: 100%; height: 150px; }
pre { background: #f0f0f0; padding: 1em; }
</style>
</head>
<body>
<h1>Markov Text Generator</h1>
<form method="post" enctype="multipart/form-data">
<p>Paste training text:</p>
<textarea name="text">{{ text }}</textarea>
<p>Or upload file: <input type="file" name="file"></p>
<p>Order: <input type="number" name="order" value="{{ order }}" min="1" max="10"></p>
<p>Mode: <select name="mode">
<option value="word" {% if mode=="word" %}selected{% endif %}>Word</option>
<option value="char" {% if mode=="char" %}selected{% endif %}>Character</option>
</select></p>
<p><input type="checkbox" name="smoothing" {% if smoothing %}checked{% endif %}> Use Laplace smoothing</p>
<p>Length: <input type="number" name="length" value="{{ length }}"></p>
<p><button type="submit">Generate</button></p>
</form>
{% if generated %}
<h2>Generated Text</h2>
<pre>{{ generated }}</pre>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    text = ""
    generated = None
    order = 2
    mode = "word"
    length = 50
    smoothing = False
    if request.method=="POST":
        text = request.form.get("text","")
        if "file" in request.files and request.files["file"].filename:
            text = request.files["file"].read().decode("utf-8")
        order = int(request.form.get("order",2))
        mode = request.form.get("mode","word")
        smoothing = bool(request.form.get("smoothing"))
        length = int(request.form.get("length",50))
        mc = MarkovChain(order=order,mode=mode,smoothing=smoothing)
        mc.train(text)
        generated = mc.generate(length=length)
    return render_template_string(TEMPLATE,text=text,order=order,mode=mode,length=length,smoothing=smoothing,generated=generated)

if __name__=="__main__":
    app.run(debug=True)
