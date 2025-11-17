MLQ Desktop Application - Package & Build Instructions (Windows EXE)

Project structure:
  mlq_desktop/
    main.py              - PyQt5 desktop application
    model_wrapper.py     - Model loader and prediction wrapper
    requirements.txt     - Python dependencies
    best_model_3_Meta_raw_data.keras             - (optional) place your Keras model file here before building
    

Quick start (development):
  1) Create a Python virtual environment and activate it.
     python -m venv venv
     venv\Scripts\activate

  2) Install dependencies:
     pip install -r requirements.txt

  3) Run the app (for testing):
     python main.py

Building Windows EXE with PyInstaller:
  1) Install PyInstaller in the same environment:
     pip install pyinstaller

  2) Use the provided batch script or run manually:
     build_exe.bat

  The batch script runs:
     pyinstaller --onefile --windowed --add-data "best_model_3_Meta_raw_data.keras;." main.py

  Notes:
    - TensorFlow is large; bundling it into a single EXE can be enormous (hundreds of MB).
      Consider shipping a small launcher EXE and installing TensorFlow as a prerequisite,
      or distribute the virtualenv with the application files.
    - If your model is a SavedModel directory, adapt the --add-data parameter to include the folder.
    - Test the EXE on a clean Windows machine to ensure all DLLs are included.
