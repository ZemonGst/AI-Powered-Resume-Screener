
# AI-Powered Resume Screener

This project uses machine learning to evaluate resumes and predict whether they match the requirements of a given job description. The backend is powered by FastAPI, and the model is trained in Google Colab.

## Project Structure

```
AI-Powered-Resume-Screener/
│── backend/              
│   ├── api.py            # FastAPI backend code
│   ├── requirements.txt  # Project dependencies
│   ├── resume_screener_model.pkl  # Trained model file
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer used for text processing
│   ├── templates/        
│   │   ├── index.html    # Frontend HTML template for user input
│   │   ├── result.html   # HTML template to show prediction results
│   ├── static/           
│   │   ├── styles.css    # CSS for frontend styling
│── dataset/              # Dataset used for training (optional)
│── model_training.ipynb  # Google Colab notebook for training the model
│── README.md             # Project documentation
│── .gitignore            # Files to ignore in version control
```

## Description

The **AI-Powered Resume Screener** uses a machine learning model to screen resumes based on their content and match them to job descriptions. The FastAPI backend serves an API that takes in a resume, processes it, and predicts how well the resume fits the job requirements.

- The model is trained using a dataset of resumes and job descriptions in Google Colab.
- The FastAPI app exposes endpoints for resume submission and prediction.
- The frontend includes a simple HTML page where users can upload resumes and view the results.

## Dependencies

You can install the necessary dependencies using the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

### Dependencies listed in `requirements.txt`:
- `fastapi`
- `uvicorn`
- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `jinja2`

## How to Run the Project

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Powered-Resume-Screener.git
cd AI-Powered-Resume-Screener
```

### 2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Run the FastAPI app:

```bash
uvicorn backend.api:app --reload
```

- This will start the server at `http://127.0.0.1:8000`.

### 4. Open the Frontend:

- Open `templates/index.html` in your browser or interact with the app through the FastAPI interface at `http://127.0.0.1:8000/docs`.

## How to Train the Model

The model is trained using the dataset of resumes and job descriptions. You can train the model yourself by running the `model_training.ipynb` notebook.

1. Open the notebook in **Google Colab**.
2. Follow the steps in the notebook to load the dataset, train the model, and save the trained model and TF-IDF vectorizer.
3. After training, the model will be saved as `resume_screener_model.pkl` and the TF-IDF vectorizer as `tfidf_vectorizer.pkl`.

Once trained, move the model files to the `backend/` folder to use them in the FastAPI app.

## Model Files

- **`resume_screener_model.pkl`**: The trained machine learning model for resume screening.
- **`tfidf_vectorizer.pkl`**: The TF-IDF vectorizer used to transform text data.

## Dataset

The dataset used for training the model consists of resumes and job descriptions. If the dataset is too large to upload, you can find it [here](provide link) or follow the instructions in the `model_training.ipynb` to download it.

## Notes

- If you plan to contribute to this project or modify the model, ensure you have the dataset and follow the instructions in the notebook.
- You can also use this system as an API by sending a POST request to the `/predict` endpoint with a resume in the form of text.
