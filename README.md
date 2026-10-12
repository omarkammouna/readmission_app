# Diabetes 30-day readmission: project demo

Student demo for the AI and Advanced Machine Learning group project. **Not a medical tool.**

The app has five pages: Home, Data and preparation, Clustering, Classification models, Try a patient.

## Run it on your computer
```
pip install -r requirements.txt
streamlit run app.py
```

## Put it online for free (Streamlit Community Cloud)
1. Create a free account on github.com, then a new **public** repository.
2. Click **Add file, Upload files**, drag in everything from this folder (including the `artifacts` folder), and commit.
3. Go to share.streamlit.io, sign in with GitHub, click **Create app**, choose your repository, the branch `main`
   and the file `app.py`.
4. Open **Advanced settings** and choose Python 3.12, then click **Deploy**.
5. After a few minutes you get a public link like `yourname.streamlit.app`.

Free apps go to sleep after 12 hours without visitors. Open the link a few minutes before your presentation.

## Files
- `app.py`: the web app.
- `artifacts/`: small result files and trained models used by the app.
- `train_models.py`: rebuilds `artifacts/` from `diabetic_data.csv` (not needed to run the app).
- `requirements.txt`: the libraries the app needs.
