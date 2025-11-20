import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

excludes = [
    'onnxscript',
    'onnx',
    'torch',
    'torchvision',
    'IPython',
    'jupyter',
    'matplotlib.tests',
    'pandas.tests',
]

hiddenimports = [
    'tensorflow',
    'keras',
    'numpy',
    'pandas',
    'matplotlib',
    'PyQt5',
]

datas = [
    ('best_model_3_Meta_raw_data.keras', '.'),     
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MLQ_Optimizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='MLQ_Optimizer'
)
