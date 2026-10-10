$ErrorActionPreference = "Stop"

python -m tkinter
python -m pip install -e ".[build]"
pyinstaller --noconfirm --clean --windowed --name PyCSR_CV --add-data "src/pycsr_cv/assets;assets" --collect-all cv2 --collect-all PIL --hidden-import pycsr_cv.concepts --hidden-import pycsr_cv.gui run_PyCSR_CV.py

Write-Host "Built dist\PyCSR_CV\PyCSR_CV.exe"
