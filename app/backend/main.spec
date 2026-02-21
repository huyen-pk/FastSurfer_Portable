# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


PROJECT_ROOT = Path('main.spec').resolve().parents[2]
FASTSURFER_HIDDENIMPORTS = collect_submodules('FastSurferCNN')
FASTSURFER_DATAS = collect_data_files('FastSurferCNN')


a = Analysis(
    ['ipc_server.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=FASTSURFER_DATAS,
    hiddenimports=FASTSURFER_HIDDENIMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='backend',
)
