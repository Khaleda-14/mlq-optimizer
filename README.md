<img width="933" height="479" alt="api" src="https://github.com/user-attachments/assets/baf6fadf-d9b2-4423-9621-633eb51bb0d3" />

MLQ Desktop Application - Package & Build Instructions (Windows EXE)

Project structure:
  mlq_optimizer/
    main.py              - PyQt5 desktop application
    model_wrapper.py     - Model loader and prediction wrapper
    requirements.txt     - Python dependencies
    best_model_3_Meta_raw_data.keras             - (optional) place your Keras model file here before building
    

Development:
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

  The batch script runs:
     pyinstaller --onefile --windowed --add-data "best_model_3_Meta_raw_data.keras;." main.py
