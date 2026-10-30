# Fixes included in this updated ZIP

1. Added `pyarrow==17.0.0`, required by Streamlit's custom component system
   used by the drawing canvas.
2. Changed the model-performance image display to `use_column_width=True`
   for compatibility with older Streamlit versions.
3. Enforced a 10 MB upload limit in the upload page.
4. Added `.streamlit/config.toml` with `server.maxUploadSize = 10`.

## Apply the update

Extract this updated ZIP into a NEW folder (or replace the old project files).
Open that folder in VS Code, activate your existing virtual environment, and run:

```powershell
python -m pip install -r requirements.txt
```

Stop the running Streamlit process first with `Ctrl+C`, then restart:

```powershell
streamlit run app.py
```

If you already have a trained model in your old project's `models` folder,
copy `digit_recognition_model.keras` into the updated project's `models` folder.
Otherwise, run `python train_model.py` once before making predictions.


## Additional compatibility fix (NumPy / PyArrow)

The earlier PyArrow pin could resolve to a build requiring NumPy 2.x. This
version pins `pyarrow==16.1.0`, which is compatible with the project's
`numpy==1.26.4` pin. In the active virtual environment, run:

```powershell
python -m pip uninstall -y pyarrow
python -m pip install -r requirements.txt
```

Stop Streamlit with Ctrl+C and restart it from the project root. The upload
page rejects files larger than 10 MB in Python. The displayed Streamlit
uploader limit is controlled by `.streamlit/config.toml`; that setting is
loaded only when the Streamlit server starts. Make sure the terminal's
current directory is the folder containing `app.py` and `.streamlit`.
