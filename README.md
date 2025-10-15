# Iris Species Prediction (SVM)

Live app: https://your-streamlit-deployment.example.com/iris-svm-app

This repository contains a small iris species prediction project using Support Vector Machines (SVM) and a Streamlit UI.

Files of interest

- `Iris (1).csv` – the CSV dataset used for training / demo.
- `train_model.py` – trains three SVM models (linear, polynomial, rbf) and saves them as `svm_linear.pkl`, `svm_poly.pkl`, and `svm_rbf.pkl`.
- `app.py` – Streamlit application. Lets you choose kernel, enter measurements and predict species. By default it loads the model files listed above.
- `requirements.txt` – Python dependencies required to run the app.

Quick start (local)

1. Create and activate a Python environment (example using venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train models (creates `svm_linear.pkl`, `svm_poly.pkl`, `svm_rbf.pkl`):

```powershell
python train_model.py
```

4. Run the Streamlit app locally:

```powershell
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501) in your browser.

Notes

- The `app.py` in this repo expects the three pickle files (`svm_linear.pkl`, `svm_poly.pkl`, `svm_rbf.pkl`) to be present in the repository root. If you prefer a single pickled pipeline or different filenames, update `app.py` accordingly.
- The "Live app" URL above is a placeholder — replace it with your actual deployed Streamlit URL if you deploy the app (Streamlit Cloud, Heroku, or other hosting).
- If you see warnings about feature names when loading models, ensure inputs are passed as a 2D array or pandas DataFrame with matching column order: [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm].

Optional improvements

- Add a `models/` directory and save models there.
- Add a small `deploy.sh` (or PowerShell script) for quick deploy steps to Streamlit Cloud.
- Add unit tests for the prediction functions.

Contact / License

This project is a demo. Use or modify as you like.
