# PyCSR_CV

**PyCSR_CV** is an interactive desktop studio for exploring computer vision and image processing. Select a concept and the app explains it in plain language, applies a real OpenCV operation to an image, and displays the result beside the input.

The package includes a sample image and a looping road-traffic video with moving cars, a searchable catalog of **50 interactive concepts across 14 learning areas**, and controls for opening and saving media. Choose a concept during video playback to see it processed live on every frame. It is designed as a visual learning aid for students, educators, and developers.

## Features

- Search concepts by name, explanation, or effect.
- Browse concepts by learning area.
- See what each selected concept means and what it does in a side help panel.
- Apply each operation immediately to the bundled sample image or an image you open.
- Play the bundled sample video on a loop, or open a local video; apply any catalog concept to each frame and view the live processed output.
- Pause and resume video playback while keeping the selected operation.
- Enumerate connected cameras, select a device, and process its live feed frame by frame.
- Compare the input and result side by side.
- Zoom both image panes in or out together, with a shared zoom level and fit reset.
- Inspect a synchronized NumPy array view for either the input or output, including shape, data type, value range, and a sampled pixel grid. The view follows still images and updates with every playing video frame. Its preview pane can be resized by dragging the divider, or hidden to give the live images more space.
- Open common image formats and save processed results.
- Run as an installed Python desktop app or package it as a Windows executable.
- Perform all image operations locally; the app does not upload images.

## Concept catalog

| Learning area | Concepts |
| --- | --- |
| Color spaces | Grayscale, HSV, CIE Lab, YCrCb, HLS |
| Thresholding | Binary, adaptive, Otsu automatic threshold |
| Filtering | Gaussian smoothing, median filtering, bilateral filtering, sharpening |
| Edges | Laplacian, Canny, horizontal and vertical Sobel, Scharr |
| Enhancement | Histogram equalization, CLAHE, invert, gamma correction, log transform |
| Morphology | Erosion, dilation, opening, closing, morphological gradient, top-hat, black-hat |
| Shapes and regions | Contour tracing |
| Features | Harris corners, Shi-Tomasi corners, ORB keypoints, SIFT keypoints, HOG descriptor |
| Frequency analysis | Fourier magnitude spectrum, DCT coefficients |
| Geometry | Resize, rotate, horizontal flip, translate, perspective warp, affine warp |
| Multi-scale | Gaussian pyramid |
| Shape analysis | Integral image, distance transform |
| Segmentation | Watershed, GrabCut foreground |
| Geometric detection | Hough line detection, Hough circle detection |

Some operations use fixed illustrative parameters. For example, the segmentation demos use automatic initial regions and may need refinement for other images. They demonstrate the underlying technique; they are not general-purpose object detectors.

## Requirements

- Python 3.9 or newer.
- A Python installation with **Tkinter** and Tcl/Tk enabled. Tkinter is included with the standard Python distribution on many platforms, but it may be an optional installation feature on Windows and Linux.
- A desktop session capable of displaying a graphical window.

Check Tkinter before installing:

```console
python -m tkinter
```

A small Tk window should appear. If Python reports `No module named tkinter`, add the Tcl/Tk feature to that Python installation, then recreate any virtual environment using that interpreter. Tkinter is provided by Python itself and is not installed from PyPI.

## Install from PyPI

After the project has been published, install it with:

```console
python -m pip install PyCSR-CV
```

The distribution name on PyPI is `pycsr-cv`. Python imports use the module name `pycsr_cv`.

Launch the GUI using either installed command:

```console
PyCSR_CV
```

```console
pycsr-cv
```

You can also launch it as a Python module:

```console
python -m pycsr_cv
```

On case-sensitive command shells, use the lowercase `pycsr-cv` command. Windows PowerShell and Command Prompt typically preserve the generated `PyCSR_CV` command spelling shown above.

## Install and run from a source checkout

Clone or download the repository, change to the project directory, and install it into a virtual environment:

```console
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
PyCSR_CV
```

macOS or Linux:

```console
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
pycsr-cv
```

For editable development, replace `python -m pip install .` with:

```console
python -m pip install -e ".[build]"
```

The optional `build` extra installs PyInstaller, used only for creating a standalone Windows app. OpenCV, NumPy, and Pillow are the runtime dependencies and are installed automatically.

## Use the application

1. Launch `PyCSR_CV` (or `pycsr-cv` on case-sensitive shells). The bundled moving-car traffic video starts automatically and loops.
2. Browse the concept list. Click a category row or its arrow to expand/collapse it; click a concept row to apply it. Categories remain open while searching and filtering.
3. Select a concept. Its meaning and effect appear in the help panel, and the live result preview applies the operation to each video frame.
4. Use **Pause video** or **Play video** to control playback. Choose **Open video** to load a local MP4, AVI, MOV, MKV, WebM, or M4V clip; opened clips also loop.
5. To use a camera, choose **Refresh cameras**, select a listed device, and click **Start camera**. The live feed is processed using the selected concept; **Stop camera** releases the device. If no devices appear, check that the camera is connected, permitted for desktop apps, and not held by another application.
6. Choose **Open image** to switch to a still JPG, JPEG, PNG, BMP, TIFF, or WebP image. The **OPEN / LOAD VIDEO** button is in the top toolbar beside **Open image**.
7. Use the **−**, **+**, and **Fit** controls in the **ZOOM BOTH IMAGE WINDOWS** toolbar just below the top toolbar. The shared zoom level shows centered details at the same scale in both image panes.
8. The **NUMPY ARRAY** pane is to the right of the input and processed result. Drag its divider to adjust its width, or use **Hide array pane** / **Show array pane** in the zoom toolbar to give the live image previews more or less space. Choose **Input array** or **Output array** from the selector in the toolbar. It displays shape, dtype, minimum and maximum values, and a sampled pixel grid with row and column coordinates. During playback it updates with each frame.
9. Select **Before / after** to compare the original and processed frames, or **Result** to return to the output view.
10. Select **Save result** to export the current frame. **Reset** stops video and restores the built-in sample image.

Selecting a different concept always applies that operation to the current input frame or image, rather than stacking it onto the previous result. During video playback, frames continue looping and the selected effect is applied independently to each frame. To chain operations, save an image result and open it again as the next input.

## Build a Windows executable

The package includes a PowerShell helper that checks Tkinter, installs the optional build tools, and creates an onedir executable:

```powershell
.\Build-PyCSR_CV.ps1
```

Or run the build commands manually from the project root:

```powershell
python -m pip install -e ".[build]"
pyinstaller --noconfirm --clean --windowed --name PyCSR_CV --add-data "src/pycsr_cv/assets;assets" --collect-all cv2 --collect-all PIL --hidden-import pycsr_cv.concepts --hidden-import pycsr_cv.gui run_PyCSR_CV.py
```

The executable and runtime files are created in `dist\PyCSR_CV\`. Launch `dist\PyCSR_CV\PyCSR_CV.exe` from that folder. `Run-PyCSR_CV.bat` starts this executable when it exists; otherwise, it tries `python -m pycsr_cv`.

Build the executable on Windows using the target Python version and architecture. PyInstaller bundles the Python runtime and dependencies, but does not remove the need to build separately for other operating systems.

## Build PyPI release artifacts

From the project root, install the packaging tools and build both standard distribution formats:

```console
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

This creates a wheel and a source distribution under `dist/`. Review both files before publishing. To publish a test release to TestPyPI first:

```console
python -m twine upload --repository testpypi dist/*
```

When the package metadata, project links, version, and release files are ready, upload to PyPI:

```console
python -m twine upload dist/*
```

PyPI authentication should be configured using an API token in the user's local `.pypirc` or entered at the prompt; do not put credentials in this repository. Increment `version` in `pyproject.toml` for each release because published PyPI release files cannot be replaced under the same version.

## Project structure

```text
PyCSR_CV/
├── pyproject.toml                 # PyPI metadata and console entry points
├── README.md
├── LICENSE
├── MANIFEST.in                    # Source distribution contents
├── requirements.txt               # Runtime dependency list
├── run_PyCSR_CV.py                # PyInstaller entry point
├── Build-PyCSR_CV.ps1             # Windows executable build helper
├── Run-PyCSR_CV.bat               # Windows launcher
└── src/
    └── pycsr_cv/
        ├── __init__.py
        ├── __main__.py
        ├── concepts.py             # Catalog and OpenCV implementations
        ├── gui.py                  # Tkinter desktop application
        └── assets/
            ├── Leena.jpg           # Bundled sample image
            └── sample_car_video.mp4 # Bundled looping road-traffic video
```

## License

PyCSR_CV is distributed under the MIT License. See [LICENSE](LICENSE).
