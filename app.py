from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import os
from markov_adv import AdvancedMarkov

app = Flask(__name__)
app.secret_key = "dev-secret"  # change for production
BASE_DIR = os.path.dirname(__file__)
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_texts")
MODEL_PATH = os.path.join(BASE_DIR, "model.json")

# a singleton model in memory for demo
MODEL = None

@app.route("/", methods=["GET", "POST"])
def index():
    global MODEL
    info = {}
    if request.method == "POST":
        action = request.form.get("action")
        mode = request.form.get("mode", "word")
        order = int(request.form.get("order", 2))
        length = int(request.form.get("length", 100))
        seed = request.form.get("seed") or None
        seed = int(seed) if seed is not None else None
        add_k = float(request.form.get("add_k") or 0.0)
        prune_min = int(request.form.get("prune_min") or 2)
        start = request.form.get("start") or None
        sample_file = request.form.get("sample_file") or None

        if action == "train":
            MODEL = AdvancedMarkov(order=order, mode=mode)
            files = []
            if sample_file and sample_file != "none":
                files.append(os.path.join(SAMPLES_DIR, sample_file))
            else:
                # train on all sample files
                for f in os.listdir(SAMPLES_DIR):
                    if f.endswith(".txt"):
                        files.append(os.path.join(SAMPLES_DIR, f))
            MODEL.train_files(files)
            MODEL.prune(min_count=prune_min)
            MODEL.save(MODEL_PATH)
            flash("Model trained and saved.", "success")
            return redirect(url_for("index"))

        elif action == "generate":
            if MODEL is None:
                if os.path.exists(MODEL_PATH):
                    MODEL = AdvancedMarkov.load(MODEL_PATH)
                else:
                    flash("No trained model found. Train first.", "danger")
                    return redirect(url_for("index"))
            start_state = None
            if start:
                if mode == "word":
                    start_state = tuple(start.split())
                else:
                    start_state = tuple(list(start))
                if len(start_state) > MODEL.order:
                    start_state = tuple(list(start_state)[-MODEL.order:])
            text = MODEL.generate(length=length, seed=seed, start_state=start_state, add_k=add_k)
            info["generated"] = text
        elif action == "perplexity":
            if MODEL is None and os.path.exists(MODEL_PATH):
                MODEL = AdvancedMarkov.load(MODEL_PATH)
            ev_text = request.form.get("eval_text") or ""
            if MODEL is None:
                flash("Train a model first.", "danger")
            else:
                ppl = MODEL.perplexity(ev_text, add_k=add_k)
                info["perplexity"] = ppl
    # list sample files
    samples = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".txt")]
    return render_template("index.html", samples=samples, info=info)

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(os.path.join(app.root_path, "sample_texts"), filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
