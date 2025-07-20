# -*- mode: python ; coding: utf-8 -*-

# Add any other large libraries you know you are not using.
# 'torch.testing' has been removed from this list to fix the "Missing Module" error.
aggressive_excludes = [
    'PyQt5', 'pandas', 'matplotlib', 
    'wandb', 'datasets', 'triton',
    'tensorboard', 'IPython', 'jupyter_client', 'jupyter_core', 'notebook'
]

a = Analysis(
    ['puckstats.pyw'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=aggressive_excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# --- One-File Executable Configuration ---
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='Puck Stats',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

# The COLLECT block is removed for a one-file build.
