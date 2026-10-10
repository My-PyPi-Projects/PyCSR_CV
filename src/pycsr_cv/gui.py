"""Tkinter-based computer vision learning studio."""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import __version__
from .concepts import BY_NAME, CATEGORIES, CONCEPTS

BG = "#0b1120"
PANEL = "#111b2e"
PANEL2 = "#17243a"
TEXT = "#e7eefb"
MUTED = "#93a4bd"
ACCENT = "#38bdf8"
GREEN = "#34d399"


def resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "assets"
    return Path(__file__).parent / "assets"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"PyCSR_CV - Computer Vision Studio {__version__}")
        root.geometry("1600x980")
        root.minsize(1180, 760)
        root.configure(bg=BG)
        self.source = self._load_sample()
        self.result = self.source.copy()
        self.video_path: Optional[Path] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.video_is_camera = False
        self.camera_devices = []
        self.camera_choice = tk.StringVar(value="Camera 0")
        self.video_after_id: Optional[str] = None
        self.video_fps = 24.0
        self.video_frame_index = 0
        self.video_playing = False
        self.video_size = (640, 360)
        self.zoom = 1.0
        self.array_target = tk.StringVar(value="Input array")
        self.array_visible = True
        self._array_refresh_id: Optional[str] = None
        self.view_mode = tk.StringVar(value="Result")
        self.current_name: Optional[str] = None
        self._photo_refs = []
        self.query = tk.StringVar()
        self.category = tk.StringVar(value="All concepts")
        self._tree_iid_to_name = {}
        self._expanded_categories = set(CATEGORIES[1:])
        self.status = tk.StringVar(value="Ready - choose any concept to explore its effect")
        self._style()
        self._build()
        self._populate()
        self._show_images()
        self.root.after(250, self.load_sample_video)

    def _load_sample(self):
        path = resource_dir() / "Leena.jpg"
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            return np.full((480, 640, 3), (42, 34, 26), np.uint8)
        return image

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 23))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 10))
        style.configure("TButton", background=PANEL2, foreground=TEXT, borderwidth=0, padding=(12, 8), font=("Segoe UI Semibold", 9))
        style.map("TButton", background=[("active", "#263b59")])
        style.configure("Accent.TButton", background="#075985", foreground="#f0f9ff")
        style.map("Accent.TButton", background=[("active", "#0369a1")])
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=TEXT, arrowcolor=ACCENT, padding=7)
        style.configure("TEntry", fieldbackground=PANEL2, foreground=TEXT, insertcolor=TEXT, padding=8)
        style.configure("TRadiobutton", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.map("TRadiobutton", foreground=[("selected", TEXT)], background=[("active", BG)])
        style.configure("Treeview", background=PANEL, foreground=TEXT, fieldbackground=PANEL, rowheight=30, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=PANEL2, foreground=MUTED, font=("Segoe UI Semibold", 9), relief="flat")
        style.map("Treeview", background=[("selected", "#075985")], foreground=[("selected", "white")])
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL2, background=ACCENT)

    def _build(self):
        header = ttk.Frame(self.root, padding=(24, 19, 24, 12)); header.pack(fill="x")
        top = ttk.Frame(header); top.pack(fill="x")
        ttk.Label(top, text="PyCSR_CV", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="COMPUTER VISION STUDIO", foreground=ACCENT, background=BG, font=("Segoe UI Semibold", 9)).pack(side="left", padx=(15, 0), pady=(10, 0))
        ttk.Label(header, text="Explore image processing and computer vision concepts interactively.", style="Subtitle.TLabel").pack(anchor="w", pady=(3, 0))

        bar = ttk.Frame(self.root, padding=(24, 0, 24, 8)); bar.pack(fill="x")
        ttk.Button(bar, text="Open image", command=self.open_image, style="Accent.TButton").pack(side="left")
        ttk.Button(bar, text="OPEN / LOAD VIDEO", command=self.open_video, style="Accent.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="Refresh cameras", command=self.refresh_cameras).pack(side="left", padx=(8, 0))
        self.camera_select = ttk.Combobox(bar, textvariable=self.camera_choice, values=("Camera 0",), state="readonly", width=11)
        self.camera_select.pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Start camera", command=self.start_camera, style="Accent.TButton").pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Reset", command=self.reset).pack(side="left", padx=8)
        ttk.Button(bar, text="Save result", command=self.save_image).pack(side="left")
        self.filename = ttk.Label(bar, text="LEENA · BUILT-IN SAMPLE", style="Muted.TLabel"); self.filename.pack(side="left", padx=12)
        self.video_controls = ttk.Frame(bar)
        self.play_button = ttk.Button(self.video_controls, text="Pause video", command=self.toggle_video)
        self.play_button.pack(side="left", padx=(0, 8))
        self.video_clock = ttk.Label(self.video_controls, text="", style="Muted.TLabel")
        self.video_clock.pack(side="left")
        self.video_controls.pack(side="left", padx=4)
        ttk.Label(bar, text="VIEW", foreground=MUTED, background=BG, font=("Segoe UI Semibold", 8)).pack(side="right", padx=(12, 0))
        ttk.Radiobutton(bar, text="Result", value="Result", variable=self.view_mode, command=self._show_images).pack(side="right", padx=4)
        ttk.Radiobutton(bar, text="Before / after", value="Before / after", variable=self.view_mode, command=self._show_images).pack(side="right", padx=4)

        zoom_bar = ttk.Frame(self.root, padding=(24, 2, 24, 10)); zoom_bar.pack(fill="x")
        ttk.Label(zoom_bar, text="ZOOM BOTH IMAGE WINDOWS", foreground=ACCENT, background=BG, font=("Segoe UI Semibold", 9)).pack(side="left", padx=(0, 10))
        ttk.Button(zoom_bar, text="−", width=3, command=lambda: self._change_zoom(0.8)).pack(side="left")
        self.zoom_label = ttk.Label(zoom_bar, text="100%", style="Muted.TLabel", width=6, anchor="center"); self.zoom_label.pack(side="left")
        ttk.Button(zoom_bar, text="+", width=3, command=lambda: self._change_zoom(1.25)).pack(side="left")
        ttk.Button(zoom_bar, text="Fit", command=self._reset_zoom).pack(side="left", padx=(6, 14))
        ttk.Label(zoom_bar, text="Both image panes zoom together", style="Muted.TLabel").pack(side="left")
        ttk.Label(zoom_bar, text="NUMPY ARRAY PANE:", foreground=MUTED, background=BG, font=("Segoe UI Semibold", 8)).pack(side="right", padx=(10, 0))
        self.array_view = ttk.Combobox(zoom_bar, textvariable=self.array_target, values=("Input array", "Output array"), state="readonly", width=15)
        self.array_view.pack(side="right")
        self.array_view.bind("<<ComboboxSelected>>", lambda _event: self._schedule_array_refresh())
        self.array_toggle = ttk.Button(zoom_bar, text="Hide array pane", command=self._toggle_array_pane)
        self.array_toggle.pack(side="right", padx=(8, 0))

        body = ttk.Frame(self.root, padding=(24, 0, 24, 10)); body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Panel.TFrame", padding=12); left.pack(side="left", fill="y"); left.configure(width=355); left.pack_propagate(False)
        ttk.Label(left, text="CONCEPT EXPLORER", style="Section.TLabel").pack(anchor="w", pady=(2, 10))
        search = ttk.Entry(left, textvariable=self.query, font=("Segoe UI", 10), style="TEntry"); search.pack(fill="x", pady=(0, 8))
        search.insert(0, "Search concepts or explanations")
        search.bind("<FocusIn>", lambda _e: search.delete(0, "end") if search.get() == "Search concepts or explanations" else None)
        search.bind("<KeyRelease>", lambda _e: self._populate())
        cat = ttk.Combobox(left, textvariable=self.category, values=CATEGORIES, state="readonly"); cat.pack(fill="x", pady=(0, 8)); cat.bind("<<ComboboxSelected>>", lambda _e: self._populate())
        self.count_label = ttk.Label(left, text="", style="Panel.TLabel", foreground=MUTED); self.count_label.pack(anchor="w", pady=(0, 5))
        tree_frame = ttk.Frame(left, style="Panel.TFrame"); tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse"); scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview); self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewOpen>>", self._category_opened)
        self.tree.bind("<<TreeviewClose>>", self._category_closed)
        self.tree.bind("<ButtonRelease-1>", self._tree_click, add="+")
        self.tree.bind("<Return>", self._tree_activate)
        self.tree.bind("<KP_Enter>", self._tree_activate)

        right = ttk.Frame(body); right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        self.help_panel = ttk.Frame(right, style="Panel.TFrame", padding=15); self.help_panel.pack(fill="x", pady=(0, 12))
        self.concept_title = ttk.Label(self.help_panel, text="Welcome to the vision lab", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 14)); self.concept_title.pack(anchor="w")
        self.concept_category = ttk.Label(self.help_panel, text="SELECT A CONCEPT TO BEGIN", background=PANEL, foreground=ACCENT, font=("Segoe UI Semibold", 8)); self.concept_category.pack(anchor="w", pady=(3, 9))
        ttk.Label(self.help_panel, text="MEANING", background=PANEL, foreground=MUTED, font=("Segoe UI Semibold", 8)).pack(anchor="w")
        self.meaning = ttk.Label(self.help_panel, text="Choose a concept in the explorer to learn the idea behind it.", style="Panel.TLabel", wraplength=900, justify="left"); self.meaning.pack(anchor="w", fill="x", pady=(3, 8))
        ttk.Label(self.help_panel, text="WHAT YOU WILL SEE", background=PANEL, foreground=MUTED, font=("Segoe UI Semibold", 8)).pack(anchor="w")
        self.effect = ttk.Label(self.help_panel, text="Your selected operation will be applied to the sample or loaded image immediately.", style="Panel.TLabel", wraplength=900, justify="left"); self.effect.pack(anchor="w", fill="x", pady=(3, 0))

        preview_split = ttk.Panedwindow(right, orient="horizontal")
        preview_split.pack(fill="both", expand=True)
        self.preview_split = preview_split
        image_area = ttk.Frame(preview_split)
        preview_split.add(image_area, weight=5)
        images = ttk.Frame(image_area); images.pack(fill="both", expand=True)
        self.image_cards = []
        for title in ("INPUT / VIDEO", "PROCESSED RESULT"):
            card = ttk.Frame(images, style="Panel.TFrame", padding=8); card.pack(side="left", fill="both", expand=True, padx=(0, 6) if len(self.image_cards)==0 else (6, 0));
            ttk.Label(card, text=title, style="Section.TLabel").pack(anchor="w", pady=(0, 7))
            label = tk.Label(card, bg="#07101e", fg=MUTED, text="", font=("Segoe UI", 11), bd=0); label.pack(fill="both", expand=True)
            self.image_cards.append((card, label))
        array_card = ttk.Frame(preview_split, style="Panel.TFrame", padding=8, width=270)
        preview_split.add(array_card, weight=1)
        array_card.pack_propagate(False)
        self.array_card = array_card
        ttk.Label(array_card, text="NUMPY ARRAY", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        array_body = ttk.Frame(array_card, style="Panel.TFrame"); array_body.pack(fill="both", expand=True)
        self.array_info = ttk.Label(array_body, text="Select an image or video frame to inspect its NumPy array.", style="Panel.TLabel", wraplength=280, justify="left")
        self.array_info.pack(anchor="w", fill="x", pady=(0, 8))
        table_frame = ttk.Frame(array_body, style="Panel.TFrame"); table_frame.pack(fill="both", expand=True)
        self.array_table = ttk.Treeview(table_frame, show="headings", height=12, style="Treeview")
        self.array_table.tag_configure("row_a", background="#132039")
        self.array_table.tag_configure("row_b", background=PANEL)
        array_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.array_table.xview)
        self.array_table.configure(xscrollcommand=array_scroll_x.set)
        self.array_table.pack(fill="both", expand=True)
        array_scroll_x.pack(fill="x")
        self.dimensions = ttk.Label(image_area, text="", style="Muted.TLabel"); self.dimensions.pack(anchor="w", pady=(9, 0))
        footer = ttk.Frame(self.root, padding=(24, 8, 24, 12)); footer.pack(fill="x")
        ttk.Separator(footer).pack(fill="x", pady=(0, 9))
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel").pack(side="left")
        ttk.Label(footer, text=f"{len(CONCEPTS)} INTERACTIVE CONCEPTS", foreground=GREEN, background=BG, font=("Segoe UI Semibold", 8)).pack(side="right")
        self.root.after_idle(self._set_initial_preview_split)

    def _set_initial_preview_split(self):
        self.root.update_idletasks()
        width = self.preview_split.winfo_width()
        if width > 0:
            self.preview_split.sashpos(0, max(400, width - 290))

    def _toggle_array_pane(self):
        if self.array_visible:
            try:
                self.preview_split.forget(self.array_card)
            except tk.TclError:
                pass
            self.array_visible = False
            self.array_toggle.configure(text="Show array pane")
        else:
            self.preview_split.add(self.array_card, weight=1)
            self.array_visible = True
            self.array_toggle.configure(text="Hide array pane")
            self.root.after_idle(self._set_initial_preview_split)

    def _populate(self):
        query = self.query.get().strip().lower(); selected = self.current_name
        for category_iid in self.tree.get_children():
            category_name = self.tree.item(category_iid, "text").split("  (")[0]
            if self.tree.item(category_iid, "open"):
                self._expanded_categories.add(category_name)
            else:
                self._expanded_categories.discard(category_name)
        self.tree.delete(*self.tree.get_children()); self._tree_iid_to_name.clear(); shown = 0
        for category in CATEGORIES[1:]:
            matches = [c for c in CONCEPTS if c.category == category and (self.category.get() == "All concepts" or category == self.category.get()) and (not query or query in c.name.lower() or query in c.meaning.lower() or query in c.effect.lower())]
            if not matches: continue
            group = self.tree.insert("", "end", iid=f"category::{category}", text=f"{category}  ({len(matches)})", open=bool(query) or category in self._expanded_categories)
            for concept in matches:
                iid = self.tree.insert(group, "end", iid=f"concept::{concept.name}", text=concept.name)
                self._tree_iid_to_name[iid] = concept.name
                shown += 1
                if concept.name == selected:
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
        self.count_label.configure(text=f"{shown} concepts - {len(CATEGORIES)-1} learning areas")

    def _category_opened(self, event=None):
        iid = self.tree.focus()
        if iid.startswith("category::"):
            self._expanded_categories.add(iid.split("::", 1)[1])

    def _category_closed(self, event=None):
        iid = self.tree.focus()
        if iid.startswith("category::"):
            self._expanded_categories.discard(iid.split("::", 1)[1])

    def _tree_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        element = self.tree.identify_element(event.x, event.y)
        self.tree.focus(iid)
        if iid.startswith("category::"):
            if "indicator" in element:
                return
            if self.tree.item(iid, "open"):
                self.tree.item(iid, open=False)
                self._expanded_categories.discard(iid.split("::", 1)[1])
            else:
                self.tree.item(iid, open=True)
                self._expanded_categories.add(iid.split("::", 1)[1])
            return "break"
        concept_name = self._tree_iid_to_name.get(iid)
        if concept_name:
            self.tree.selection_set(iid)
            self._show_concept(concept_name)
            return "break"

    def _tree_activate(self, _event=None):
        iid = self.tree.focus()
        if iid.startswith("category::"):
            self.tree.item(iid, open=not self.tree.item(iid, "open"))
            category = iid.split("::", 1)[1]
            if self.tree.item(iid, "open"):
                self._expanded_categories.add(category)
            else:
                self._expanded_categories.discard(category)
            return "break"
        concept_name = self._tree_iid_to_name.get(iid)
        if concept_name:
            self._show_concept(concept_name)
            return "break"

    def _selected(self, _event=None):
        selection = self.tree.selection()
        if not selection: return
        name = self._tree_iid_to_name.get(selection[0])
        if name:
            self._show_concept(name)

    def _show_concept(self, name):
        if name not in BY_NAME:
            return
        concept = BY_NAME[name]; self.current_name = name
        self.concept_title.configure(text=concept.name); self.concept_category.configure(text=f"{concept.category.upper()}  ·  {concept.params or 'DEFAULT PARAMETERS'}")
        self.meaning.configure(text=concept.meaning); self.effect.configure(text=concept.effect)
        if self.video_capture is not None:
            self.status.set(f"Live video: applying {concept.name} to each frame")
            self._schedule_video_frame(0)
            return
        try:
            self.result = concept.operation(self.source.copy())
            self.status.set(f"Applied {concept.name} - {self.result.shape[1]} x {self.result.shape[0]} result")
        except Exception as exc:
            self.result = self.source.copy(); self.status.set(f"Could not apply {concept.name}: {exc}")
        self._show_images()

    def _fit(self, image, size):
        h, w = image.shape[:2]; bound_w = max(100, size[0]-28); bound_h = max(100, size[1]-58); fit_scale=min(bound_w/w,bound_h/h)
        scale=fit_scale*self.zoom
        if self.zoom > 1.0:
            crop_w=max(1,min(w,int(bound_w/scale)))
            crop_h=max(1,min(h,int(bound_h/scale)))
            x0=(w-crop_w)//2; y0=(h-crop_h)//2
            image=image[y0:y0+crop_h,x0:x0+crop_w]
            h,w=image.shape[:2]
            scale=min(bound_w/w,bound_h/h)
        image=cv2.resize(image,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        if image.ndim==2: image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else: image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    def _show_images(self):
        if self.video_capture is not None:
            return
        self.root.update_idletasks(); mode=self.view_mode.get()
        sources = [self.source, self.result]
        if mode == "Before / after": sources = [self.source, self.result]
        refs=[]
        for idx, (card,label) in enumerate(self.image_cards):
            card.winfo_children()[0].configure(text="INPUT IMAGE" if idx==0 else "PROCESSED RESULT")
            try:
                pic=self._fit(sources[idx],(max(300,card.winfo_width()),max(260,card.winfo_height())))
                photo=ImageTk.PhotoImage(pic); label.configure(image=photo,text=""); refs.append(photo)
            except Exception: pass
        self._photo_refs=refs
        self.dimensions.configure(text=f"Input  {self.source.shape[1]} × {self.source.shape[0]} px     ·     Output  {self.result.shape[1]} × {self.result.shape[0]} px     ·     Select another concept to compare")
        self._schedule_array_refresh()

    def _change_zoom(self, factor: float):
        self.zoom = min(4.0, max(0.25, self.zoom * factor))
        self.zoom_label.configure(text=f"{round(self.zoom * 100)}%")
        if self.video_capture is None:
            self._show_images()

    def _reset_zoom(self):
        self.zoom = 1.0
        self.zoom_label.configure(text="100%")
        if self.video_capture is None:
            self._show_images()

    def _schedule_array_refresh(self):
        if self._array_refresh_id is not None:
            self.root.after_cancel(self._array_refresh_id)
        self._array_refresh_id = self.root.after(25, self._refresh_array_view)

    def _refresh_array_view(self):
        self._array_refresh_id = None
        array = self.source if self.array_target.get() == "Input array" else self.result
        if array is None:
            return
        array = np.asarray(array)
        h, w = array.shape[:2]
        row_indexes = np.linspace(0, h - 1, min(4, h), dtype=int)
        col_indexes = np.linspace(0, w - 1, min(6, w), dtype=int)
        sample = array[np.ix_(row_indexes, col_indexes)]
        info = f"{self.array_target.get()}  |  shape={array.shape}  |  dtype={array.dtype}  |  min={array.min()}  max={array.max()}  |  showing sampled top-left {sample.shape[0]} x {sample.shape[1]} values"
        self.array_info.configure(text=info)
        channels = array.shape[2] if array.ndim == 3 else 1
        headings = [f"x={x}" for x in col_indexes[:sample.shape[1]]]
        self.array_table.configure(columns=headings, displaycolumns=headings)
        for col in headings:
            self.array_table.heading(col, text=col)
            self.array_table.column(col, width=max(90, 90 * channels), minwidth=80, stretch=True, anchor="center")
        self.array_table.delete(*self.array_table.get_children())
        for row_index, row in enumerate(sample):
            values = []
            for pixel in row:
                if np.ndim(pixel) == 0:
                    values.append(str(int(pixel)))
                else:
                    values.append("(" + ", ".join(str(int(value)) for value in pixel) + ")")
            self.array_table.insert("", "end", values=values, tags=("row_a" if row_index % 2 == 0 else "row_b",))

    def open_image(self):
        path=filedialog.askopenfilename(title="Open an image",filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),("All files","*.*")])
        if not path: return
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None: messagebox.showerror("Could not open image","Choose a valid image file."); return
        self._close_video()
        self._show_images()
        self.source=img; self.result=img.copy(); self.view_mode.set("Result"); self.filename.configure(text=Path(path).name.upper()); self.status.set(f"Loaded {Path(path).name} - select a concept to apply it")
        if self.current_name: self._selected()
        else: self._show_images()

    def _sample_video_path(self) -> Path:
        bundled = resource_dir() / "sample_car_video.mp4"
        if bundled.is_file():
            return bundled
        raise FileNotFoundError("The packaged sample video is missing. Reinstall PyCSR_CV or open a video file.")

    def load_sample_video(self):
        try:
            self._start_video(self._sample_video_path(), "SAMPLE CAR VIDEO · LOOPING")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Sample video unavailable", str(exc))

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
        if path:
            try:
                self._start_video(Path(path), Path(path).name.upper())
            except (OSError, ValueError) as exc:
                messagebox.showerror("Could not open video", str(exc))

    def refresh_cameras(self):
        self.status.set("Checking camera devices...")
        self.root.update_idletasks()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        devices = []
        for index in range(8):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _frame = cap.read()
                cap.release()
                if ok:
                    devices.append(index)
            else:
                cap.release()
        self.camera_devices = devices
        choices = [f"Camera {index}" for index in devices]
        self.camera_select.configure(values=choices or ["No cameras found"])
        if choices:
            if self.camera_choice.get() not in choices:
                self.camera_choice.set(choices[0])
            self.status.set(f"Found {len(choices)} camera device(s)")
        else:
            self.camera_choice.set("No cameras found")
            self.status.set("No camera devices found - check permissions and connections")

    def start_camera(self):
        choice = self.camera_choice.get()
        if not choice.startswith("Camera "):
            self.refresh_cameras()
            choice = self.camera_choice.get()
        try:
            index = int(choice.split()[1])
        except (IndexError, ValueError):
            messagebox.showerror("No camera selected", "Refresh the device list and select an available camera.")
            return
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Camera unavailable", f"Could not open Camera {index}. It may be in use by another application.")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, first = capture.read()
        if not ok or first is None:
            capture.release()
            messagebox.showerror("Camera unavailable", f"Camera {index} opened but returned no video frames.")
            return
        self._close_video()
        self.video_capture = capture
        self.video_path = None
        self.video_is_camera = True
        self.video_fps = 30.0
        self.video_frame_index = 0
        self.source = first
        self.result = first.copy()
        self.filename.configure(text=f"LIVE CAMERA {index}")
        self.video_controls.pack(side="left", padx=(4, 0))
        self.play_button.configure(text="Stop camera", command=self.stop_camera)
        self.video_clock.configure(text="LIVE")
        self.video_playing = True
        self.view_mode.set("Result")
        self.status.set(f"Live feed from Camera {index} - choose a concept to process each frame")
        self.concept_title.configure(text="Live camera mode")
        self.concept_category.configure(text=f"CAMERA {index}")
        self.meaning.configure(text="Live frames are captured from the selected camera device.")
        self.effect.configure(text="Choose any concept to apply its OpenCV operation to each live frame. Use Stop camera to release the device.")
        self._schedule_video_frame(0)

    def stop_camera(self):
        if not self.video_is_camera:
            return
        self._close_video()
        self.play_button.configure(text="Pause video", command=self.toggle_video)
        self.filename.configure(text="CAMERA STOPPED")
        self.status.set("Camera stopped - select a video or image, or start a camera again")
        self.source = self._load_sample()
        self.result = self.source.copy()
        self._show_images()

    def _start_video(self, path: Path, label: str):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"OpenCV could not read this video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = capture.read()
        if not ok or first is None:
            capture.release()
            raise ValueError("The video has no readable frames.")
        self._close_video()
        self.video_capture = capture
        self.video_path = path
        self.video_is_camera = False
        self.video_fps = fps if np.isfinite(fps) and 1.0 <= fps <= 120.0 else 24.0
        self.video_frame_index = 0
        self.video_size = (first.shape[1], first.shape[0])
        self.source = first
        self.result = first.copy()
        self.filename.configure(text=label)
        self.video_controls.pack(side="left", padx=(4, 0))
        self.play_button.configure(text="Pause video")
        self.video_playing = True
        self.view_mode.set("Result")
        self.status.set("Looping video - choose a concept to process every frame")
        self.concept_title.configure(text="Video learning mode")
        self.concept_category.configure(text="SAMPLE CAR VIDEO" if path.name == "sample_car_video.mp4" else "VIDEO INPUT")
        self.meaning.configure(text="The selected video is playing continuously. Choose any concept to apply its OpenCV operation independently to every frame.")
        self.effect.configure(text="The left pane shows the current source frame. The result pane updates live with the selected concept; changing concepts switches the effect on the next frame.")
        duration = frame_count / self.video_fps if frame_count > 0 else 0.0
        self.video_clock.configure(text=f"0:00 / {int(duration // 60)}:{int(duration % 60):02d}" if duration else "Looping")
        self._schedule_video_frame(0)

    def _close_video(self):
        if self.video_after_id is not None:
            try:
                self.root.after_cancel(self.video_after_id)
            except tk.TclError:
                pass
            self.video_after_id = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_path = None
        self.video_is_camera = False
        self.video_playing = False
        if hasattr(self, "video_controls"):
            self.video_controls.pack_forget()
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Pause video", command=self.toggle_video)
        if hasattr(self, "image_cards"):
            for index, (card, _label) in enumerate(self.image_cards):
                card.winfo_children()[0].configure(text="INPUT IMAGE" if index == 0 else "PROCESSED RESULT")

    def toggle_video(self):
        if self.video_capture is None:
            return
        self.video_playing = not self.video_playing
        self.play_button.configure(text="Pause video" if self.video_playing else "Play video")
        if self.video_playing:
            self._schedule_video_frame(0)
        elif self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None

    def _schedule_video_frame(self, delay: Optional[int] = None):
        if self.video_capture is None or not self.video_playing:
            return
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
        interval = max(10, int(1000 / self.video_fps)) if delay is None else delay
        self.video_after_id = self.root.after(interval, self._next_video_frame)

    def _next_video_frame(self):
        self.video_after_id = None
        capture = self.video_capture
        if capture is None or not self.video_playing:
            return
        ok, frame = capture.read()
        if (not ok or frame is None) and self.video_is_camera:
            self.status.set("Camera feed stopped - the device is no longer returning frames")
            self._close_video()
            self.play_button.configure(text="Pause video", command=self.toggle_video)
            return
        if not ok or frame is None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_frame_index = 0
            ok, frame = capture.read()
        if not ok or frame is None:
            self.status.set("Video playback stopped because no frames could be decoded")
            self.video_playing = False
            self.play_button.configure(text="Play video")
            return
        self.video_frame_index += 1
        self.source = frame
        if self.current_name and self.current_name in BY_NAME:
            try:
                self.result = BY_NAME[self.current_name].operation(frame.copy())
            except Exception as exc:
                self.result = frame.copy()
                self.status.set(f"Could not process this frame: {exc}")
        else:
            self.result = frame.copy()
        self._render_video_frame(frame, self.result)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / self.video_fps if count > 0 else 0.0
        seconds = self.video_frame_index / self.video_fps
        if duration:
            position = seconds % duration
            self.video_clock.configure(text=f"{int(position // 60)}:{int(position % 60):02d} / {int(duration // 60)}:{int(duration % 60):02d}")
        else:
            self.video_clock.configure(text="Looping")
        self._schedule_video_frame()

    def _render_video_frame(self, original: np.ndarray, processed: np.ndarray):
        self.root.update_idletasks()
        images = (original, processed)
        refs = []
        for index, (card, label) in enumerate(self.image_cards):
            card.winfo_children()[0].configure(text="LIVE VIDEO FRAME" if index == 0 else "LIVE PROCESSED RESULT")
            try:
                photo = ImageTk.PhotoImage(self._fit(images[index], (max(300, card.winfo_width()), max(260, card.winfo_height()))))
                label.configure(image=photo, text="")
                refs.append(photo)
            except (tk.TclError, ValueError):
                pass
        self._photo_refs = refs
        self.dimensions.configure(text=f"Live frame {self.video_frame_index} - {original.shape[1]} x {original.shape[0]} px - loop playback")
        self._schedule_array_refresh()

    def reset(self):
        self._close_video()
        self.play_button.configure(text="Pause video", command=self.toggle_video)
        self.source=self._load_sample(); self.result=self.source.copy(); self.current_name=None; self.filename.configure(text="LEENA · BUILT-IN SAMPLE")
        self.concept_title.configure(text="Welcome to the vision lab"); self.concept_category.configure(text="SELECT A CONCEPT TO BEGIN")
        self.meaning.configure(text="Choose a concept in the explorer to learn the idea behind it."); self.effect.configure(text="Your selected operation will be applied to the sample or loaded image immediately.")
        self.view_mode.set("Result"); self.status.set("Reset to built-in sample image"); self._show_images()

    def on_close(self):
        self._close_video()
        if self._array_refresh_id is not None:
            try:
                self.root.after_cancel(self._array_refresh_id)
            except tk.TclError:
                pass
            self._array_refresh_id = None
        self.root.destroy()

    def save_image(self):
        path=filedialog.asksaveasfilename(title="Save processed result",defaultextension=".png",filetypes=[("PNG image","*.png"),("JPEG image","*.jpg"),("Bitmap image","*.bmp")],initialfile="PyCSR_CV_result.png")
        if path:
            try:
                if not cv2.imwrite(path,self.result): raise OSError("The image format could not be written.")
                self.status.set(f"Saved result to {path}")
            except Exception as exc: messagebox.showerror("Could not save image",str(exc))


def main():
    root=tk.Tk(); app=App(root); root.protocol("WM_DELETE_WINDOW", app.on_close); root.mainloop()


if __name__ == "__main__":
    main()
