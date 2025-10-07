# PRODIGY_GA_03
📝 Advanced Markov Text Generator with Flask

This project is an enhanced Markov text generator that combines traditional Markov chain modeling with advanced features, all wrapped in a user-friendly Flask web interface. Users can train models, generate text, and evaluate model performance interactively or via Python scripts.

📂 Project Structure
app.py – Flask web application for training, generating, and evaluating text.
markov_adv.py – Contains the AdvancedMarkov class with advanced Markov features.
sample_texts/ – Folder containing sample text files for training.
model.json – Serialized model saved after training.
⚙️ Key Features
Flask Web UI: Train models, generate text, and view evaluation metrics via a browser.
Advanced Markov Enhancements (markov_adv.py):
Add-k (Laplace) smoothing to handle unseen transitions
Backoff to lower-order models for better predictions
Pruning of low-count transitions to optimize memory
Perplexity calculation to evaluate model performance
Flexible Training: Train on a single file or multiple samples in sample_texts/.
CLI & Script Usage: Supports methods like train_files, generate, perplexity, prune, save, and load.
🛠️ Installation & Setup
Create a virtual environment (optional but recommended):
python -m venv .venv

Activate the virtual environment:
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

Install Flask:
pip install Flask

🚀 Running the Flask App
Start the Flask server:
python app.py

Open your browser and navigate to:
http://127.0.0.1:5000

Use the web interface to train the model, generate text, and evaluate performance.
🖥️ CLI / Script Usage

You can also interact with the model programmatically via Python:

from markov_adv import AdvancedMarkov

model = AdvancedMarkov()
model.train_files(['sample_texts/file1.txt', 'sample_texts/file2.txt'])
model.generate(length=100)
model.perplexity(['sample_texts/test.txt'])
model.prune(min_count=2)
model.save('model.json')
model.load('model.json')

⚠️ Notes
This implementation is suitable for demo and small-scale text corpora.
For large datasets, consider streaming training and memory optimization.
The trained model is saved as model.json in the project root for reuse.
💡 Learnings & Highlights
Built a web interface with Flask for AI/ML model interaction.
Implemented advanced Markov techniques like smoothing, backoff, and pruning.
Learned text generation evaluation with perplexity metrics.
Gained experience in combining backend ML models with interactive frontends.

This project is a practical demonstration of how classic NLP methods can be enhanced and deployed in a user-friendly, interactive environment.
