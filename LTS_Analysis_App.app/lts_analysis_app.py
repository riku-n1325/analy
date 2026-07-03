from __future__ import annotations

import csv
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_density import (  # noqa: E402
    PAPER_N2_STOKES_CROSS_SECTION_M2,
    PAPER_THOMSON_CROSS_SECTION_M2,
    electron_density_from_thomson_counts,
    electron_temperature_from_width,
    gas_density_from_pressure,
    scattering_parameter_alpha,
    throughput_from_raman,
)
from fit_peak import fit_gaussian_log_parabola, fit_multiple_gaussian_peaks  # noqa: E402
from fit_thomson import fit_broad_gaussian  # noqa: E402
from read_spe import read_spe, save_image_csv, save_spe_like, spectrum_from_image  # noqa: E402


APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / "app_settings.json"


def write_summary(path: Path, values: dict[str, tuple[float, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit"])
        for name, (value, unit) in values.items():
            writer.writerow([name, value, unit])


def write_fit_curve(
    path: Path,
    raw: np.ndarray,
    smooth: np.ndarray,
    baseline: np.ndarray,
    fit_curve: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel", "raw_counts", "smoothed_counts", "baseline_counts", "fit_counts", "residual_counts"])
        for pixel, values in enumerate(zip(raw, smooth, baseline, fit_curve)):
            raw_value, smooth_value, baseline_value, fit_value = values
            writer.writerow(
                [
                    pixel,
                    float(raw_value),
                    float(smooth_value),
                    float(baseline_value),
                    float(fit_value),
                    float(smooth_value - fit_value),
                ]
            )


class LtsAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LTS ラマン校正トムソン散乱解析")
        self.geometry("1100x760")
        self.minsize(980, 680)

        self.thomson_path = tk.StringVar()
        default_raman = APP_DIR / "virtual_raman.spe"
        self.raman_path = tk.StringVar(value=str(default_raman) if default_raman.exists() else "")
        self.raman_start_path = tk.StringVar(value=str(default_raman) if default_raman.exists() else "")
        self.raman_end_path = tk.StringVar(value=str(default_raman) if default_raman.exists() else "")
        self.out_dir = tk.StringVar(value=str(APP_DIR))

        self.y_min = tk.StringVar(value="600")
        self.y_max = tk.StringVar(value="970")
        self.fixed_center = tk.StringVar(value="536.8749")
        self.raman_center = tk.StringVar(value="650")
        self.raman_stokes_min = tk.StringVar(value="")
        self.raman_stokes_max = tk.StringVar(value="")
        self.raman_max_peaks = tk.StringVar(value="12")
        self.raman_peak_threshold = tk.StringVar(value="0.08")
        self.nm_per_pixel = tk.StringVar(value="0.021")
        self.laser_wavelength_nm = tk.StringVar(value="532")
        self.scattering_angle_deg = tk.StringVar(value="90")
        self.instrument_sigma_pixel = tk.StringVar(value="0")
        self.ts_fit_min = tk.StringVar(value="360")
        self.ts_fit_max = tk.StringVar(value="760")
        self.ts_baseline_left_min = tk.StringVar(value="180")
        self.ts_baseline_left_max = tk.StringVar(value="330")
        self.ts_baseline_right_min = tk.StringVar(value="800")
        self.ts_baseline_right_max = tk.StringVar(value="1000")
        self.ts_median_width = tk.StringVar(value="21")
        self.ts_threshold_fraction = tk.StringVar(value="0.15")
        self.ts_fix_center = tk.BooleanVar(value=True)

        self.pressure_pa = tk.StringVar(value="1000")
        self.gas_temperature_k = tk.StringVar(value="300")
        self.raman_dsigma = tk.StringVar(value=f"{PAPER_N2_STOKES_CROSS_SECTION_M2:.3e}")
        self.thomson_dsigma = tk.StringVar(value=f"{PAPER_THOMSON_CROSS_SECTION_M2:.3e}")
        self.raman_shots = tk.StringVar(value="1")
        self.thomson_shots = tk.StringVar(value="1")
        self.raman_energy_mj = tk.StringVar(value="1")
        self.thomson_energy_mj = tk.StringVar(value="1")
        self.calibration_mode = tk.StringVar(value="single")
        self.thomson_time_h = tk.StringVar(value="")
        self.raman_start_time_h = tk.StringVar(value="0")
        self.raman_end_time_h = tk.StringVar(value="")
        self.raman_start_pressure_pa = tk.StringVar(value="1000")
        self.raman_end_pressure_pa = tk.StringVar(value="1000")
        self.raman_start_shots = tk.StringVar(value="1")
        self.raman_end_shots = tk.StringVar(value="1")
        self.raman_start_energy_mj = tk.StringVar(value="1")
        self.raman_end_energy_mj = tk.StringVar(value="1")
        self.correction_factor = tk.StringVar(value="1")
        self.area_kind = tk.StringVar(value="gaussian")

        self.pressure_folder = tk.StringVar(value="")
        self.pressure_peak_center = tk.StringVar(value="")
        self.pressure_signal_kind = tk.StringVar(value="amplitude")
        self.pressure_rows: list[tuple[Path, tk.StringVar]] = []

        self.subtract_out_dir = tk.StringVar(value=str(APP_DIR))
        self.subtract_scale = tk.StringVar(value="1")
        self.subtract_save_csv = tk.BooleanVar(value=True)
        self.subtract_save_spe = tk.BooleanVar(value=True)
        self.subtract_target_files: list[Path] = []
        self.subtract_background_files: list[Path] = []
        self.subtract_image_photo: ImageTk.PhotoImage | None = None
        self.subtract_fit_kind = tk.StringVar(value="thomson")

        self.viewer_path = tk.StringVar()
        self.viewer_x_min = tk.StringVar(value="")
        self.viewer_x_max = tk.StringVar(value="")
        self.viewer_y_min = tk.StringVar(value="")
        self.viewer_y_max = tk.StringVar(value="")
        self.viewer_image_photo: ImageTk.PhotoImage | None = None
        self.viewer_current_image: np.ndarray | None = None
        self.viewer_display: dict[str, float | tuple[int, int]] = {}
        self.viewer_drag_start: tuple[int, int] | None = None
        self.viewer_drag_rect_id: int | None = None
        self._pending_pressure_rows: list[dict[str, str]] = []
        self._pending_geometry = ""

        self._load_settings()
        self._build_ui()
        self._restore_pressure_rows()
        self._refresh_subtract_file_lists()
        if self._pending_geometry:
            try:
                self.geometry(self._pending_geometry)
            except tk.TclError:
                pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self._build_calibration_tab(notebook)
        self._build_pressure_tab(notebook)
        self._build_subtraction_tab(notebook)
        self._build_viewer_tab(notebook)

    def _settings_vars(self) -> dict[str, tk.StringVar]:
        return {
            "thomson_path": self.thomson_path,
            "raman_path": self.raman_path,
            "raman_start_path": self.raman_start_path,
            "raman_end_path": self.raman_end_path,
            "out_dir": self.out_dir,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "fixed_center": self.fixed_center,
            "raman_center": self.raman_center,
            "raman_stokes_min": self.raman_stokes_min,
            "raman_stokes_max": self.raman_stokes_max,
            "raman_max_peaks": self.raman_max_peaks,
            "raman_peak_threshold": self.raman_peak_threshold,
            "nm_per_pixel": self.nm_per_pixel,
            "laser_wavelength_nm": self.laser_wavelength_nm,
            "scattering_angle_deg": self.scattering_angle_deg,
            "instrument_sigma_pixel": self.instrument_sigma_pixel,
            "ts_fit_min": self.ts_fit_min,
            "ts_fit_max": self.ts_fit_max,
            "ts_baseline_left_min": self.ts_baseline_left_min,
            "ts_baseline_left_max": self.ts_baseline_left_max,
            "ts_baseline_right_min": self.ts_baseline_right_min,
            "ts_baseline_right_max": self.ts_baseline_right_max,
            "ts_median_width": self.ts_median_width,
            "ts_threshold_fraction": self.ts_threshold_fraction,
            "pressure_pa": self.pressure_pa,
            "gas_temperature_k": self.gas_temperature_k,
            "raman_dsigma": self.raman_dsigma,
            "thomson_dsigma": self.thomson_dsigma,
            "raman_shots": self.raman_shots,
            "thomson_shots": self.thomson_shots,
            "raman_energy_mj": self.raman_energy_mj,
            "thomson_energy_mj": self.thomson_energy_mj,
            "calibration_mode": self.calibration_mode,
            "thomson_time_h": self.thomson_time_h,
            "raman_start_time_h": self.raman_start_time_h,
            "raman_end_time_h": self.raman_end_time_h,
            "raman_start_pressure_pa": self.raman_start_pressure_pa,
            "raman_end_pressure_pa": self.raman_end_pressure_pa,
            "raman_start_shots": self.raman_start_shots,
            "raman_end_shots": self.raman_end_shots,
            "raman_start_energy_mj": self.raman_start_energy_mj,
            "raman_end_energy_mj": self.raman_end_energy_mj,
            "correction_factor": self.correction_factor,
            "area_kind": self.area_kind,
            "pressure_folder": self.pressure_folder,
            "pressure_peak_center": self.pressure_peak_center,
            "pressure_signal_kind": self.pressure_signal_kind,
            "subtract_out_dir": self.subtract_out_dir,
            "subtract_scale": self.subtract_scale,
            "subtract_fit_kind": self.subtract_fit_kind,
            "viewer_path": self.viewer_path,
            "viewer_x_min": self.viewer_x_min,
            "viewer_x_max": self.viewer_x_max,
            "viewer_y_min": self.viewer_y_min,
            "viewer_y_max": self.viewer_y_max,
        }

    def _settings_bools(self) -> dict[str, tk.BooleanVar]:
        return {
            "subtract_save_csv": self.subtract_save_csv,
            "subtract_save_spe": self.subtract_save_spe,
            "ts_fix_center": self.ts_fix_center,
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        values = data.get("values", {})
        if isinstance(values, dict):
            for name, var in self._settings_vars().items():
                if name in values:
                    var.set(str(values[name]))

        bools = data.get("booleans", {})
        if isinstance(bools, dict):
            for name, var in self._settings_bools().items():
                if name in bools:
                    var.set(bool(bools[name]))

        self.subtract_target_files = [Path(path) for path in data.get("subtract_target_files", []) if path]
        self.subtract_background_files = [Path(path) for path in data.get("subtract_background_files", []) if path]
        rows = data.get("pressure_rows", [])
        if isinstance(rows, list):
            self._pending_pressure_rows = [row for row in rows if isinstance(row, dict)]
        self._pending_geometry = str(data.get("geometry", ""))

    def _save_settings(self) -> None:
        data = {
            "values": {name: var.get() for name, var in self._settings_vars().items()},
            "booleans": {name: bool(var.get()) for name, var in self._settings_bools().items()},
            "subtract_target_files": [str(path) for path in self.subtract_target_files],
            "subtract_background_files": [str(path) for path in self.subtract_background_files],
            "pressure_rows": [
                {"path": str(path), "pressure": pressure_var.get()}
                for path, pressure_var in self.pressure_rows
            ],
            "geometry": self.geometry(),
        }
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _on_close(self) -> None:
        try:
            self._save_settings()
        except Exception as exc:
            messagebox.showwarning("設定保存エラー", f"入力値を保存できませんでした。\n{exc}")
        self.destroy()

    def _build_calibration_tab(self, notebook: ttk.Notebook) -> None:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="ラマン校正TS解析")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        canvas = tk.Canvas(tab, highlightthickness=0)
        y_scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        x_scroll = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        root = ttk.Frame(canvas, padding=12)
        window_id = canvas.create_window((0, 0), window=root, anchor="nw")

        def update_scrollregion(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def update_inner_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=max(event.width, root.winfo_reqwidth()))
            update_scrollregion()

        def on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        root.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", update_inner_width)
        canvas.bind("<Enter>", lambda _event: canvas.bind_all("<MouseWheel>", on_mousewheel))
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        file_box = ttk.LabelFrame(root, text="ファイル", padding=10)
        file_box.grid(row=0, column=0, sticky="ew")
        file_box.columnconfigure(1, weight=1)
        self._file_row(file_box, 0, "トムソンSPE", self.thomson_path)
        self._file_row(file_box, 1, "ラマンSPE", self.raman_path)
        self._file_row(file_box, 2, "開始ラマンSPE", self.raman_start_path)
        self._file_row(file_box, 3, "終了ラマンSPE", self.raman_end_path)
        self._dir_row(file_box, 4, "保存先", self.out_dir)

        param_box = ttk.LabelFrame(root, text="解析条件", padding=10)
        param_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for i in range(6):
            param_box.columnconfigure(i, weight=1)

        self._entry(param_box, 0, 0, "TS y最小", self.y_min, "pixel")
        self._entry(param_box, 0, 1, "TS y最大", self.y_max, "pixel")
        self._entry(param_box, 0, 2, "レーザー中心", self.fixed_center, "pixel")
        self._entry(param_box, 0, 3, "ラマン中心", self.raman_center, "pixel")

        self._entry(param_box, 1, 0, "波長校正", self.nm_per_pixel, "nm/pixel")
        self._entry(param_box, 1, 1, "レーザー波長", self.laser_wavelength_nm, "nm")
        self._entry(param_box, 1, 2, "散乱角", self.scattering_angle_deg, "deg")
        self._entry(param_box, 1, 3, "装置幅sigma", self.instrument_sigma_pixel, "pixel")
        self._entry(param_box, 1, 4, "水素圧力", self.pressure_pa, "Pa")
        self._entry(param_box, 1, 5, "気体温度", self.gas_temperature_k, "K")

        self._entry(param_box, 2, 0, "ラマン有効断面積", self.raman_dsigma, "m^2")
        self._entry(param_box, 2, 1, "TS断面積", self.thomson_dsigma, "m^2")
        self._entry(param_box, 2, 2, "ラマンshot数", self.raman_shots, "shots")
        self._entry(param_box, 2, 3, "TS shot数", self.thomson_shots, "shots")
        self._entry(param_box, 2, 4, "ラマンEnergy", self.raman_energy_mj, "mJ")
        self._entry(param_box, 2, 5, "TS Energy", self.thomson_energy_mj, "mJ")
        self._entry(param_box, 3, 0, "Stokes探索最小", self.raman_stokes_min, "pixel")
        self._entry(param_box, 3, 1, "Stokes探索最大", self.raman_stokes_max, "pixel")
        self._entry(param_box, 3, 2, "最大ピーク数", self.raman_max_peaks, "個")
        self._entry(param_box, 3, 3, "ピークしきい値", self.raman_peak_threshold, "-")
        self._entry(param_box, 3, 4, "TS時刻", self.thomson_time_h, "h")
        self._entry(param_box, 3, 5, "補正係数", self.correction_factor, "-")
        self._entry(param_box, 4, 0, "開始時刻", self.raman_start_time_h, "h")
        self._entry(param_box, 4, 1, "開始圧力", self.raman_start_pressure_pa, "Pa")
        self._entry(param_box, 4, 2, "開始shot数", self.raman_start_shots, "shots")
        self._entry(param_box, 4, 3, "開始Energy", self.raman_start_energy_mj, "mJ")
        self._entry(param_box, 5, 0, "終了時刻", self.raman_end_time_h, "h")
        self._entry(param_box, 5, 1, "終了圧力", self.raman_end_pressure_pa, "Pa")
        self._entry(param_box, 5, 2, "終了shot数", self.raman_end_shots, "shots")
        self._entry(param_box, 5, 3, "終了Energy", self.raman_end_energy_mj, "mJ")

        area_frame = ttk.Frame(param_box)
        area_frame.grid(row=6, column=0, columnspan=6, sticky="w", pady=(8, 0))
        ttk.Label(area_frame, text="校正").pack(side="left")
        ttk.Radiobutton(area_frame, text="単一ラマン", value="single", variable=self.calibration_mode).pack(side="left", padx=8)
        ttk.Radiobutton(area_frame, text="時間補間", value="drift", variable=self.calibration_mode).pack(side="left")
        ttk.Label(area_frame, text="面積").pack(side="left")
        ttk.Radiobutton(area_frame, text="ガウス面積", value="gaussian", variable=self.area_kind).pack(side="left", padx=8)
        ttk.Radiobutton(area_frame, text="直接積分", value="direct", variable=self.area_kind).pack(side="left")
        ttk.Radiobutton(area_frame, text="複数Stokes合計", value="multi_stokes", variable=self.area_kind).pack(side="left", padx=8)

        actions = ttk.Frame(root)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="解析", command=self.analyze).pack(side="left")
        ttk.Button(actions, text="ログ消去", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=8)

        self.log = tk.Text(root, height=20, wrap="word")
        self.log.grid(row=3, column=0, sticky="nsew", pady=(10, 0))

    def _build_pressure_tab(self, notebook: ttk.Notebook) -> None:
        root = ttk.Frame(notebook, padding=12)
        notebook.add(root, text="ラマン圧力依存性")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        folder_box = ttk.LabelFrame(root, text="スペクトルフォルダ", padding=10)
        folder_box.grid(row=0, column=0, sticky="ew")
        folder_box.columnconfigure(1, weight=1)
        ttk.Label(folder_box, text="フォルダ").grid(row=0, column=0, sticky="w")
        ttk.Entry(folder_box, textvariable=self.pressure_folder).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(folder_box, text="選択", command=self._browse_pressure_folder).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(folder_box, text="SPE一覧読込", command=self._load_pressure_folder).grid(row=0, column=3)

        option_box = ttk.LabelFrame(root, text="解析条件", padding=10)
        option_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for i in range(5):
            option_box.columnconfigure(i, weight=1)
        self._entry(option_box, 0, 0, "ラマン中心", self.pressure_peak_center, "pixel")

        signal_frame = ttk.Frame(option_box)
        signal_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Label(signal_frame, text="縦軸").pack(anchor="w")
        ttk.Radiobutton(signal_frame, text="ピーク高さ", value="amplitude", variable=self.pressure_signal_kind).pack(side="left")
        ttk.Radiobutton(signal_frame, text="ガウス面積", value="area", variable=self.pressure_signal_kind).pack(side="left", padx=8)

        body = ttk.Frame(root)
        body.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        list_box = ttk.LabelFrame(body, text="圧力入力", padding=8)
        list_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        list_box.rowconfigure(1, weight=1)
        list_box.columnconfigure(0, weight=1)
        ttk.Label(list_box, text="各SPEファイルの測定圧力を入力してください。").grid(row=0, column=0, sticky="w")

        self.pressure_canvas = tk.Canvas(list_box, height=280)
        pressure_scroll = ttk.Scrollbar(list_box, orient="vertical", command=self.pressure_canvas.yview)
        self.pressure_table = ttk.Frame(self.pressure_canvas)
        self.pressure_table.bind(
            "<Configure>",
            lambda _event: self.pressure_canvas.configure(scrollregion=self.pressure_canvas.bbox("all")),
        )
        self.pressure_canvas.create_window((0, 0), window=self.pressure_table, anchor="nw")
        self.pressure_canvas.configure(yscrollcommand=pressure_scroll.set)
        self.pressure_canvas.grid(row=1, column=0, sticky="nsew")
        pressure_scroll.grid(row=1, column=1, sticky="ns")

        action_frame = ttk.Frame(list_box)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(action_frame, text="解析", command=self.analyze_pressure_dependence).pack(side="left")
        ttk.Button(action_frame, text="入力クリア", command=self._clear_pressure_rows).pack(side="left", padx=8)

        graph_box = ttk.LabelFrame(body, text="圧力依存性グラフ", padding=8)
        graph_box.grid(row=0, column=1, sticky="nsew")
        graph_box.rowconfigure(0, weight=1)
        graph_box.columnconfigure(0, weight=1)
        self.pressure_plot = tk.Canvas(graph_box, bg="white", height=420)
        self.pressure_plot.grid(row=0, column=0, sticky="nsew")
        self.pressure_result = tk.Text(graph_box, height=8, wrap="word")
        self.pressure_result.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_subtraction_tab(self, notebook: ttk.Notebook) -> None:
        root = ttk.Frame(notebook, padding=12)
        notebook.add(root, text="スペクトル差し引き")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)
        root.rowconfigure(3, weight=1)

        top_box = ttk.LabelFrame(root, text="保存条件", padding=10)
        top_box.grid(row=0, column=0, sticky="ew")
        top_box.columnconfigure(1, weight=1)
        self._dir_row(top_box, 0, "保存先", self.subtract_out_dir)
        self._entry(top_box, 0, 3, "差し引き倍率", self.subtract_scale, "-")
        ttk.Checkbutton(top_box, text="CSV保存", variable=self.subtract_save_csv).grid(row=0, column=4, padx=8)
        ttk.Checkbutton(top_box, text="SPE保存", variable=self.subtract_save_spe).grid(row=0, column=5, padx=8)

        body = ttk.Frame(root)
        body.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        target_box = ttk.LabelFrame(body, text="差し引き対象SPE", padding=8)
        target_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        target_box.columnconfigure(0, weight=1)
        target_box.rowconfigure(0, weight=1)
        self.subtract_target_list = tk.Listbox(target_box, height=14, selectmode="extended")
        self.subtract_target_list.grid(row=0, column=0, sticky="nsew")
        target_scroll = ttk.Scrollbar(target_box, orient="vertical", command=self.subtract_target_list.yview)
        target_scroll.grid(row=0, column=1, sticky="ns")
        self.subtract_target_list.configure(yscrollcommand=target_scroll.set)
        target_actions = ttk.Frame(target_box)
        target_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(target_actions, text="SPE追加", command=self._add_subtract_targets).pack(side="left")
        ttk.Button(target_actions, text="選択削除", command=lambda: self._remove_selected_subtract("target")).pack(side="left", padx=8)
        ttk.Button(target_actions, text="全削除", command=lambda: self._clear_subtract_files("target")).pack(side="left")

        bg_box = ttk.LabelFrame(body, text="差し引くSPE（複数の場合は平均）", padding=8)
        bg_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        bg_box.columnconfigure(0, weight=1)
        bg_box.rowconfigure(0, weight=1)
        self.subtract_background_list = tk.Listbox(bg_box, height=14, selectmode="extended")
        self.subtract_background_list.grid(row=0, column=0, sticky="nsew")
        bg_scroll = ttk.Scrollbar(bg_box, orient="vertical", command=self.subtract_background_list.yview)
        bg_scroll.grid(row=0, column=1, sticky="ns")
        self.subtract_background_list.configure(yscrollcommand=bg_scroll.set)
        bg_actions = ttk.Frame(bg_box)
        bg_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(bg_actions, text="SPE追加", command=self._add_subtract_backgrounds).pack(side="left")
        ttk.Button(bg_actions, text="選択削除", command=lambda: self._remove_selected_subtract("background")).pack(side="left", padx=8)
        ttk.Button(bg_actions, text="全削除", command=lambda: self._clear_subtract_files("background")).pack(side="left")

        preview_box = ttk.LabelFrame(root, text="差し引き後プレビュー", padding=8)
        preview_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        preview_box.columnconfigure(0, weight=1)
        preview_box.columnconfigure(1, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self.subtract_image_canvas = tk.Canvas(preview_box, bg="white", height=180)
        self.subtract_image_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.subtract_spectrum_canvas = tk.Canvas(preview_box, bg="white", height=180)
        self.subtract_spectrum_canvas.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        fit_box = ttk.LabelFrame(root, text="スペクトルフィッティング", padding=8)
        fit_box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        fit_box.columnconfigure(0, weight=1)
        fit_box.columnconfigure(1, weight=1)
        fit_box.rowconfigure(1, weight=1)

        fit_controls = ttk.Frame(fit_box)
        fit_controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(fit_controls, text="種別").pack(side="left")
        ttk.Radiobutton(fit_controls, text="トムソン", value="thomson", variable=self.subtract_fit_kind).pack(side="left", padx=8)
        ttk.Radiobutton(fit_controls, text="ラマン", value="raman", variable=self.subtract_fit_kind).pack(side="left")
        ttk.Button(fit_controls, text="フィット実行", command=self.fit_subtracted_spectrum).pack(side="left", padx=12)

        ts_fit_box = ttk.Frame(fit_controls)
        ts_fit_box.pack(side="left", fill="x", expand=True)
        self._entry(ts_fit_box, 0, 0, "TS fit最小", self.ts_fit_min, "pixel")
        self._entry(ts_fit_box, 0, 1, "TS fit最大", self.ts_fit_max, "pixel")
        self._entry(ts_fit_box, 0, 2, "背景左最小", self.ts_baseline_left_min, "pixel")
        self._entry(ts_fit_box, 0, 3, "背景左最大", self.ts_baseline_left_max, "pixel")
        self._entry(ts_fit_box, 0, 4, "背景右最小", self.ts_baseline_right_min, "pixel")
        self._entry(ts_fit_box, 0, 5, "背景右最大", self.ts_baseline_right_max, "pixel")
        self._entry(ts_fit_box, 1, 0, "平滑化幅", self.ts_median_width, "pixel")
        self._entry(ts_fit_box, 1, 1, "fitしきい値", self.ts_threshold_fraction, "-")
        self._entry(ts_fit_box, 1, 2, "TS y最小", self.y_min, "pixel")
        self._entry(ts_fit_box, 1, 3, "TS y最大", self.y_max, "pixel")
        ttk.Checkbutton(ts_fit_box, text="レーザー中心を固定", variable=self.ts_fix_center).grid(
            row=1,
            column=4,
            columnspan=2,
            sticky="w",
            padx=4,
            pady=3,
        )

        self.subtract_fit_canvas = tk.Canvas(fit_box, bg="white", height=260)
        self.subtract_fit_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(8, 0))
        self.subtract_residual_canvas = tk.Canvas(fit_box, bg="white", height=260)
        self.subtract_residual_canvas.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(8, 0))

        bottom = ttk.Frame(root)
        bottom.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(bottom, text="プレビュー更新", command=self.preview_subtraction).pack(side="left")
        ttk.Button(bottom, text="差し引き保存", command=self.analyze_subtraction).pack(side="left")
        ttk.Button(bottom, text="ログ消去", command=lambda: self.subtract_log.delete("1.0", "end")).pack(side="left", padx=8)

        self.subtract_log = tk.Text(root, height=10, wrap="word")
        self.subtract_log.grid(row=5, column=0, sticky="ew", pady=(10, 0))

    def _build_viewer_tab(self, notebook: ttk.Notebook) -> None:
        root = ttk.Frame(notebook, padding=12)
        notebook.add(root, text="SPEビューア")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        file_box = ttk.LabelFrame(root, text="ファイルと積算範囲", padding=10)
        file_box.grid(row=0, column=0, sticky="ew")
        file_box.columnconfigure(1, weight=1)
        self._file_row(file_box, 0, "SPEファイル", self.viewer_path)
        self._entry(file_box, 0, 3, "x最小", self.viewer_x_min, "pixel")
        self._entry(file_box, 0, 4, "x最大", self.viewer_x_max, "pixel")
        self._entry(file_box, 1, 3, "y最小", self.viewer_y_min, "pixel")
        self._entry(file_box, 1, 4, "y最大", self.viewer_y_max, "pixel")
        ttk.Button(file_box, text="表示", command=self.show_spe_viewer).grid(row=0, column=5, padx=8)
        ttk.Button(file_box, text="解析タブへ反映", command=self.apply_viewer_roi_to_analysis).grid(row=1, column=5, padx=8)

        view_box = ttk.Frame(root)
        view_box.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        view_box.columnconfigure(0, weight=1)
        view_box.columnconfigure(1, weight=1)
        view_box.rowconfigure(0, weight=1)

        image_box = ttk.LabelFrame(view_box, text="SPE画像", padding=8)
        image_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        image_box.columnconfigure(0, weight=1)
        image_box.rowconfigure(0, weight=1)
        self.viewer_image_canvas = tk.Canvas(image_box, bg="white", height=430)
        self.viewer_image_canvas.grid(row=0, column=0, sticky="nsew")
        self.viewer_image_canvas.bind("<ButtonPress-1>", self._viewer_drag_start)
        self.viewer_image_canvas.bind("<B1-Motion>", self._viewer_drag_motion)
        self.viewer_image_canvas.bind("<ButtonRelease-1>", self._viewer_drag_end)

        spectrum_box = ttk.LabelFrame(view_box, text="横方向スペクトル", padding=8)
        spectrum_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        spectrum_box.columnconfigure(0, weight=1)
        spectrum_box.rowconfigure(0, weight=1)
        self.viewer_spectrum_canvas = tk.Canvas(spectrum_box, bg="white", height=430)
        self.viewer_spectrum_canvas.grid(row=0, column=0, sticky="nsew")

        self.viewer_log = tk.Text(root, height=7, wrap="word")
        self.viewer_log.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _file_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="選択", command=lambda: self._browse_file(var)).grid(row=row, column=2, pady=3)

    def _dir_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="選択", command=lambda: self._browse_dir(var)).grid(row=row, column=2, pady=3)

    def _entry(self, parent: ttk.Frame, row: int, col: int, label: str, var: tk.StringVar, unit: str) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, sticky="ew", padx=4, pady=3)
        ttk.Label(frame, text=label).pack(anchor="w")
        value_frame = ttk.Frame(frame)
        value_frame.pack(fill="x")
        ttk.Entry(value_frame, textvariable=var, width=12).pack(side="left", fill="x", expand=True)
        ttk.Label(value_frame, text=unit, width=9).pack(side="left", padx=(4, 0))

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _add_subtract_targets(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        for path_text in paths:
            path = Path(path_text)
            if path not in self.subtract_target_files:
                self.subtract_target_files.append(path)
                self.subtract_target_list.insert("end", path.name)

    def _add_subtract_backgrounds(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        for path_text in paths:
            path = Path(path_text)
            if path not in self.subtract_background_files:
                self.subtract_background_files.append(path)
                self.subtract_background_list.insert("end", path.name)

    def _remove_selected_subtract(self, kind: str) -> None:
        if kind == "target":
            listbox = self.subtract_target_list
            files = self.subtract_target_files
        else:
            listbox = self.subtract_background_list
            files = self.subtract_background_files
        for index in reversed(list(listbox.curselection())):
            listbox.delete(index)
            del files[index]

    def _clear_subtract_files(self, kind: str) -> None:
        if kind == "target":
            self.subtract_target_list.delete(0, "end")
            self.subtract_target_files.clear()
        else:
            self.subtract_background_list.delete(0, "end")
            self.subtract_background_files.clear()

    def _refresh_subtract_file_lists(self) -> None:
        self.subtract_target_list.delete(0, "end")
        for path in self.subtract_target_files:
            self.subtract_target_list.insert("end", path.name)
        self.subtract_background_list.delete(0, "end")
        for path in self.subtract_background_files:
            self.subtract_background_list.insert("end", path.name)

    def _populate_pressure_rows(self, files: list[Path], pressures: dict[str, str] | None = None) -> None:
        pressures = pressures or {}
        for child in self.pressure_table.winfo_children():
            child.destroy()
        self.pressure_rows.clear()
        self.pressure_result.delete("1.0", "end")
        self.pressure_plot.delete("all")

        if not files:
            return

        ttk.Label(self.pressure_table, text="ファイル名", width=36).grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Label(self.pressure_table, text="圧力", width=12).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(self.pressure_table, text="単位", width=8).grid(row=0, column=2, sticky="w", padx=2, pady=2)

        for row, path in enumerate(files, start=1):
            pressure_var = tk.StringVar(value=pressures.get(str(path), pressures.get(path.name, "")))
            ttk.Label(self.pressure_table, text=path.name, width=42).grid(row=row, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(self.pressure_table, textvariable=pressure_var, width=14).grid(row=row, column=1, sticky="w", padx=2, pady=2)
            ttk.Label(self.pressure_table, text="Pa", width=8).grid(row=row, column=2, sticky="w", padx=2, pady=2)
            self.pressure_rows.append((path, pressure_var))

    def _restore_pressure_rows(self) -> None:
        if not self._pending_pressure_rows:
            return
        files: list[Path] = []
        pressures: dict[str, str] = {}
        for row in self._pending_pressure_rows:
            path_text = str(row.get("path", ""))
            if not path_text:
                continue
            path = Path(path_text)
            files.append(path)
            pressures[str(path)] = str(row.get("pressure", ""))
        self._populate_pressure_rows(files, pressures)

    def _float(self, var: tk.StringVar, name: str) -> float:
        try:
            return float(var.get())
        except ValueError as exc:
            raise ValueError(f"{name} は数値で入力してください: {var.get()}") from exc

    def _int(self, var: tk.StringVar, name: str) -> int:
        try:
            return int(float(var.get()))
        except ValueError as exc:
            raise ValueError(f"{name} は整数で入力してください: {var.get()}") from exc

    def _optional_float(self, var: tk.StringVar, name: str) -> float | None:
        text = var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name} は数値または空欄で入力してください: {var.get()}") from exc

    def _optional_int(self, var: tk.StringVar, name: str) -> int | None:
        value = self._optional_float(var, name)
        if value is None:
            return None
        return int(round(value))

    @staticmethod
    def _scaled_photo(image: np.ndarray, canvas: tk.Canvas) -> ImageTk.PhotoImage:
        arr = image.astype(np.float64)
        lo, hi = np.percentile(arr, [1, 99.7])
        if hi <= lo:
            hi = float(arr.max() if arr.max() > lo else lo + 1)
        scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
        pil_image = Image.fromarray((scaled * 255).astype(np.uint8), mode="L")

        canvas.update_idletasks()
        max_w = max(canvas.winfo_width(), 320)
        max_h = max(canvas.winfo_height(), 180)
        scale = min(max_w / pil_image.width, max_h / pil_image.height)
        new_size = (max(1, int(pil_image.width * scale)), max(1, int(pil_image.height * scale)))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_image)

    def _draw_image_canvas(
        self,
        canvas: tk.Canvas,
        image: np.ndarray,
        title: str,
        attr_name: str,
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        canvas.delete("all")
        photo = self._scaled_photo(image, canvas)
        setattr(self, attr_name, photo)
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 320)
        height = max(canvas.winfo_height(), 180)
        image_x0 = (width - photo.width()) / 2
        image_y0 = (height - photo.height()) / 2
        canvas.create_image(width / 2, height / 2, image=photo, anchor="center")
        if roi is not None:
            x0, x1, y0, y1 = roi
            img_h, img_w = image.shape
            scale_x = photo.width() / img_w
            scale_y = photo.height() / img_h
            rx0 = image_x0 + x0 * scale_x
            rx1 = image_x0 + x1 * scale_x
            ry0 = image_y0 + y0 * scale_y
            ry1 = image_y0 + y1 * scale_y
            canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="#d34a35", width=2)
            canvas.create_text(rx0 + 4, ry0 + 10, anchor="w", text=f"x={x0}:{x1}, y={y0}:{y1}", fill="#d34a35")
        canvas.create_text(image_x0, min(height - 8, image_y0 + photo.height() + 12), anchor="w", text="x pixel", fill="black")
        canvas.create_text(max(8, image_x0 - 10), image_y0, anchor="e", text="y=0", fill="black")
        canvas.create_text(image_x0 + photo.width(), min(height - 8, image_y0 + photo.height() + 12), anchor="e", text=f"x={image.shape[1]-1}", fill="black")
        canvas.create_text(max(8, image_x0 - 10), image_y0 + photo.height(), anchor="e", text=f"y={image.shape[0]-1}", fill="black")
        canvas.create_rectangle(0, 0, width, 24, fill="white", outline="")
        canvas.create_text(8, 12, anchor="w", text=title, fill="black")
        if attr_name == "viewer_image_photo":
            self.viewer_display = {
                "image_x0": image_x0,
                "image_y0": image_y0,
                "scale_x": photo.width() / image.shape[1],
                "scale_y": photo.height() / image.shape[0],
                "shape": image.shape,
            }

    @staticmethod
    def _draw_spectrum_canvas(
        canvas: tk.Canvas,
        spectrum: np.ndarray,
        title: str,
        y_label: str = "counts",
        x_start: int = 0,
    ) -> None:
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 180)
        left, right, top, bottom = 62, 20, 28, 48
        plot_w = width - left - right
        plot_h = height - top - bottom
        arr = spectrum.astype(np.float64)
        x_min, x_max = float(x_start), float(x_start + max(arr.size - 1, 1))
        y_min = float(np.percentile(arr, 1))
        y_max = float(np.percentile(arr, 99.7))
        if y_max <= y_min:
            y_max = float(arr.max() if arr.max() > y_min else y_min + 1)
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        def px(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def py(y: float) -> float:
            return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

        canvas.create_text(left, 12, anchor="w", text=title)
        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="black")
        canvas.create_line(left, top, left, top + plot_h, fill="black")
        canvas.create_text(left + plot_w / 2, height - 18, text="pixel")
        canvas.create_text(18, top + plot_h / 2, text=y_label, angle=90)

        for i in range(5):
            x_val = x_min + (x_max - x_min) * i / 4
            x_pos = px(x_val)
            canvas.create_line(x_pos, top + plot_h, x_pos, top + plot_h + 4, fill="black")
            canvas.create_text(x_pos, top + plot_h + 16, text=f"{x_val:.0f}")
            y_val = y_min + (y_max - y_min) * i / 4
            y_pos = py(y_val)
            canvas.create_line(left - 4, y_pos, left, y_pos, fill="black")
            canvas.create_text(left - 8, y_pos, text=f"{y_val:.2g}", anchor="e")

        if arr.size < 2:
            return
        step = max(1, int(arr.size / plot_w))
        indices = np.arange(0, arr.size, step)
        xs = indices + x_start
        ys = arr[indices]
        points = []
        for x_value, y_value in zip(xs, ys):
            points.extend([px(float(x_value)), py(float(y_value))])
        if len(points) >= 4:
            canvas.create_line(*points, fill="#1f5fbf", width=1)

    @staticmethod
    def _draw_fit_overlay_canvas(
        canvas: tk.Canvas,
        x_values: np.ndarray,
        data_values: np.ndarray,
        fit_values: np.ndarray,
        title: str,
    ) -> None:
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 420)
        height = max(canvas.winfo_height(), 220)
        left, right, top, bottom = 62, 22, 30, 48
        plot_w = width - left - right
        plot_h = height - top - bottom
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        if x_max <= x_min:
            x_max = x_min + 1.0
        all_y = np.concatenate([data_values.astype(np.float64), fit_values.astype(np.float64)])
        y_min = float(np.percentile(all_y, 1))
        y_max = float(np.percentile(all_y, 99.7))
        if y_max <= y_min:
            y_max = float(np.max(all_y) if np.max(all_y) > y_min else y_min + 1)
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        def px(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def py(y: float) -> float:
            return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

        canvas.create_text(left, 12, anchor="w", text=title)
        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="black")
        canvas.create_line(left, top, left, top + plot_h, fill="black")
        canvas.create_text(left + plot_w / 2, height - 18, text="pixel")
        canvas.create_text(18, top + plot_h / 2, text="counts", angle=90)
        canvas.create_text(width - 130, 14, text="blue: data  red: fit", fill="black")

        for i in range(5):
            x_val = x_min + (x_max - x_min) * i / 4
            x_pos = px(x_val)
            canvas.create_line(x_pos, top + plot_h, x_pos, top + plot_h + 4, fill="black")
            canvas.create_text(x_pos, top + plot_h + 16, text=f"{x_val:.0f}")
            y_val = y_min + (y_max - y_min) * i / 4
            y_pos = py(y_val)
            canvas.create_line(left - 4, y_pos, left, y_pos, fill="black")
            canvas.create_text(left - 8, y_pos, text=f"{y_val:.2g}", anchor="e")

        def draw_line(values: np.ndarray, color: str, width_px: int) -> None:
            points = []
            for x_value, y_value in zip(x_values, values):
                points.extend([px(float(x_value)), py(float(y_value))])
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=width_px)

        draw_line(data_values, "#1f5fbf", 1)
        draw_line(fit_values, "#d34a35", 2)

    def _fit_image_for_subtraction_tab(self) -> tuple[Path, np.ndarray]:
        if self.subtract_background_files:
            return self._make_subtracted_images(preview_only=True)[0]
        if not self.subtract_target_files:
            raise ValueError("差し引き対象SPEを1つ以上追加してください。")
        if self.subtract_target_list.curselection():
            path = self.subtract_target_files[self.subtract_target_list.curselection()[0]]
        else:
            path = self.subtract_target_files[0]
        spe = read_spe(path)
        return path, spe.image.astype(np.float64)

    def _make_subtracted_images(self, preview_only: bool = False) -> list[tuple[Path, np.ndarray]]:
        if not self.subtract_target_files:
            raise ValueError("差し引き対象SPEを1つ以上追加してください。")
        if not self.subtract_background_files:
            raise ValueError("差し引くSPEを1つ以上追加してください。")

        scale = self._float(self.subtract_scale, "差し引き倍率")
        background_images = []
        reference_shape = None
        for bg_path in self.subtract_background_files:
            bg = read_spe(bg_path)
            image = bg.image.astype(np.float64)
            if reference_shape is None:
                reference_shape = image.shape
            elif image.shape != reference_shape:
                raise ValueError(
                    f"差し引くSPEの画像サイズが一致しません: {bg_path.name} "
                    f"{image.shape} != {reference_shape}"
                )
            background_images.append(image)

        background_mean = np.mean(np.stack(background_images, axis=0), axis=0)
        target_files = self.subtract_target_files
        if preview_only and self.subtract_target_list.curselection():
            target_files = [self.subtract_target_files[self.subtract_target_list.curselection()[0]]]
        elif preview_only:
            target_files = [self.subtract_target_files[0]]

        results = []
        for target_path in target_files:
            target = read_spe(target_path)
            target_image = target.image.astype(np.float64)
            if target_image.shape != background_mean.shape:
                raise ValueError(
                    f"対象SPEと差し引くSPEの画像サイズが一致しません: {target_path.name} "
                    f"{target_image.shape} != {background_mean.shape}"
                )
            results.append((target_path, target_image - scale * background_mean))
        return results

    def _viewer_roi(self, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
        height, width = image_shape
        x_min = self._optional_int(self.viewer_x_min, "x最小")
        x_max = self._optional_int(self.viewer_x_max, "x最大")
        y_min = self._optional_int(self.viewer_y_min, "y最小")
        y_max = self._optional_int(self.viewer_y_max, "y最大")
        x0 = 0 if x_min is None else x_min
        x1 = width if x_max is None else x_max
        y0 = 0 if y_min is None else y_min
        y1 = height if y_max is None else y_max
        if not (0 <= x0 < x1 <= width):
            raise ValueError(f"x範囲が不正です: {x0}:{x1} for width {width}")
        if not (0 <= y0 < y1 <= height):
            raise ValueError(f"y範囲が不正です: {y0}:{y1} for height {height}")
        return x0, x1, y0, y1

    def _viewer_canvas_to_pixel(self, canvas_x: int, canvas_y: int) -> tuple[int, int]:
        if not self.viewer_display:
            raise ValueError("先にSPE画像を表示してください。")
        shape = self.viewer_display.get("shape")
        if not isinstance(shape, tuple):
            raise ValueError("表示中の画像情報を取得できません。")
        height, width = shape
        image_x0 = float(self.viewer_display["image_x0"])
        image_y0 = float(self.viewer_display["image_y0"])
        scale_x = float(self.viewer_display["scale_x"])
        scale_y = float(self.viewer_display["scale_y"])
        x = int(round((canvas_x - image_x0) / scale_x))
        y = int(round((canvas_y - image_y0) / scale_y))
        return min(max(x, 0), width), min(max(y, 0), height)

    def _viewer_drag_start(self, event: tk.Event) -> None:
        try:
            self.viewer_drag_start = self._viewer_canvas_to_pixel(int(event.x), int(event.y))
            if self.viewer_drag_rect_id is not None:
                self.viewer_image_canvas.delete(self.viewer_drag_rect_id)
                self.viewer_drag_rect_id = None
        except Exception:
            self.viewer_drag_start = None

    def _viewer_drag_motion(self, event: tk.Event) -> None:
        if self.viewer_drag_start is None or not self.viewer_display:
            return
        try:
            x0, y0 = self.viewer_drag_start
            x1, y1 = self._viewer_canvas_to_pixel(int(event.x), int(event.y))
            image_x0 = float(self.viewer_display["image_x0"])
            image_y0 = float(self.viewer_display["image_y0"])
            scale_x = float(self.viewer_display["scale_x"])
            scale_y = float(self.viewer_display["scale_y"])
            rx0 = image_x0 + x0 * scale_x
            rx1 = image_x0 + x1 * scale_x
            ry0 = image_y0 + y0 * scale_y
            ry1 = image_y0 + y1 * scale_y
            if self.viewer_drag_rect_id is not None:
                self.viewer_image_canvas.delete(self.viewer_drag_rect_id)
            self.viewer_drag_rect_id = self.viewer_image_canvas.create_rectangle(
                rx0,
                ry0,
                rx1,
                ry1,
                outline="#d34a35",
                width=2,
                dash=(4, 2),
            )
        except Exception:
            return

    def _viewer_drag_end(self, event: tk.Event) -> None:
        if self.viewer_drag_start is None:
            return
        try:
            sx, sy = self.viewer_drag_start
            ex, ey = self._viewer_canvas_to_pixel(int(event.x), int(event.y))
            x0, x1 = sorted((sx, ex))
            y0, y1 = sorted((sy, ey))
            if x1 <= x0:
                x1 = x0 + 1
            if y1 <= y0:
                y1 = y0 + 1
            if self.viewer_display and isinstance(self.viewer_display.get("shape"), tuple):
                height, width = self.viewer_display["shape"]
                x0 = min(max(x0, 0), width - 1)
                x1 = min(max(x1, x0 + 1), width)
                y0 = min(max(y0, 0), height - 1)
                y1 = min(max(y1, y0 + 1), height)
            self.viewer_x_min.set(str(x0))
            self.viewer_x_max.set(str(x1))
            self.viewer_y_min.set(str(y0))
            self.viewer_y_max.set(str(y1))
            self.viewer_drag_start = None
            self.show_spe_viewer()
        except Exception as exc:
            self.viewer_drag_start = None
            messagebox.showerror("ROI指定エラー", str(exc))

    def apply_viewer_roi_to_analysis(self) -> None:
        try:
            path = Path(self.viewer_path.get())
            if not path.exists():
                raise FileNotFoundError(f"SPEファイルが見つかりません: {path}")
            spe = read_spe(path)
            self.viewer_current_image = spe.image
            x0, x1, y0, y1 = self._viewer_roi(spe.image.shape)
            self.y_min.set(str(y0))
            self.y_max.set(str(y1))
            self.ts_fit_min.set(str(x0))
            self.ts_fit_max.set(str(x1))
            self.raman_stokes_min.set(str(x0))
            self.raman_stokes_max.set(str(x1))
            self.viewer_log.insert(
                "end",
                f"解析タブへ反映しました: TS y={y0}:{y1}, TS fit x={x0}:{x1}, Stokes探索 x={x0}:{x1}\n",
            )
            self.viewer_log.see("end")
        except Exception as exc:
            messagebox.showerror("反映エラー", str(exc))
            self.viewer_log.insert("end", f"エラー: {exc}\n")

    def _fit_raman_signal(
        self,
        raman_file: Path,
        raman_center: float | None,
        raman_stokes_min: int | None,
        raman_stokes_max: int | None,
        raman_max_peaks: int,
        raman_peak_threshold: float,
    ) -> dict[str, object]:
        raman = read_spe(raman_file)
        raman_spectrum = spectrum_from_image(raman.image, None, None)
        raman_multi_fit = None
        if self.area_kind.get() == "multi_stokes":
            raman_multi_fit = fit_multiple_gaussian_peaks(
                raman_spectrum,
                min_pixel=raman_stokes_min,
                max_pixel=raman_stokes_max,
                max_peaks=raman_max_peaks,
                window=14,
                sideband=55,
                min_prominence_fraction=raman_peak_threshold,
            )
            raman_fit = raman_multi_fit.peaks[0]
            area = raman_multi_fit.total_gaussian_area
            r_squared = raman_multi_fit.mean_r_squared
            peak_count = len(raman_multi_fit.peaks)
            peak_pixels = ", ".join(f"{peak.center_pixel:.2f}" for peak in raman_multi_fit.peaks)
        else:
            raman_fit = fit_gaussian_log_parabola(
                raman_spectrum,
                peak_pixel=None if raman_center is None else int(round(raman_center)),
                window=45,
                sideband=100,
                fixed_center=raman_center,
            )
            if self.area_kind.get() == "gaussian":
                area = raman_fit.gaussian_area
            else:
                area = raman_fit.direct_area
            r_squared = raman_fit.r_squared
            peak_count = 1
            peak_pixels = f"{raman_fit.center_pixel:.2f}"

        return {
            "fit": raman_fit,
            "area": float(area),
            "r_squared": float(r_squared),
            "peak_count": int(peak_count),
            "peak_pixels": str(peak_pixels),
        }

    @staticmethod
    def _positive(value: float, name: str) -> float:
        if value <= 0:
            raise ValueError(f"{name} は正の値を入力してください: {value}")
        return value

    def analyze(self) -> None:
        try:
            out_dir = Path(self.out_dir.get())
            out_dir.mkdir(parents=True, exist_ok=True)

            thomson_file = Path(self.thomson_path.get())
            raman_file = Path(self.raman_path.get())
            raman_start_file = Path(self.raman_start_path.get())
            raman_end_file = Path(self.raman_end_path.get())
            if not thomson_file.exists():
                raise FileNotFoundError(f"トムソンSPEが見つかりません: {thomson_file}")
            if self.calibration_mode.get() == "single" and not raman_file.exists():
                raise FileNotFoundError(f"ラマンSPEが見つかりません: {raman_file}")
            if self.calibration_mode.get() == "drift":
                if not raman_start_file.exists():
                    raise FileNotFoundError(f"開始ラマンSPEが見つかりません: {raman_start_file}")
                if not raman_end_file.exists():
                    raise FileNotFoundError(f"終了ラマンSPEが見つかりません: {raman_end_file}")

            y_min = self._int(self.y_min, "TS y最小")
            y_max = self._int(self.y_max, "TS y最大")
            center = self._float(self.fixed_center, "レーザー中心")
            raman_center = self._optional_float(self.raman_center, "ラマン中心")
            raman_stokes_min = self._optional_int(self.raman_stokes_min, "Stokes探索最小")
            raman_stokes_max = self._optional_int(self.raman_stokes_max, "Stokes探索最大")
            raman_max_peaks = self._int(self.raman_max_peaks, "最大ピーク数")
            raman_peak_threshold = self._float(self.raman_peak_threshold, "ピークしきい値")
            nm_per_pixel = self._float(self.nm_per_pixel, "波長校正")
            laser_nm = self._float(self.laser_wavelength_nm, "レーザー波長")
            angle = self._float(self.scattering_angle_deg, "散乱角")
            inst_sigma = self._float(self.instrument_sigma_pixel, "装置幅sigma")
            pressure = self._float(self.pressure_pa, "水素圧力")
            gas_temp = self._float(self.gas_temperature_k, "気体温度")
            raman_dsigma = self._float(self.raman_dsigma, "ラマン有効断面積")
            thomson_dsigma = self._float(self.thomson_dsigma, "TS断面積")
            raman_shots = self._float(self.raman_shots, "ラマンshot数")
            thomson_shots = self._float(self.thomson_shots, "TS shot数")
            raman_energy_mj = self._float(self.raman_energy_mj, "ラマンEnergy")
            thomson_energy_mj = self._float(self.thomson_energy_mj, "TS Energy")
            correction = self._float(self.correction_factor, "補正係数")
            ts_fit_min = self._int(self.ts_fit_min, "TS fit最小")
            ts_fit_max = self._int(self.ts_fit_max, "TS fit最大")
            ts_baseline_left_min = self._int(self.ts_baseline_left_min, "背景左最小")
            ts_baseline_left_max = self._int(self.ts_baseline_left_max, "背景左最大")
            ts_baseline_right_min = self._int(self.ts_baseline_right_min, "背景右最小")
            ts_baseline_right_max = self._int(self.ts_baseline_right_max, "背景右最大")
            ts_median_width = self._int(self.ts_median_width, "平滑化幅")
            ts_threshold_fraction = self._float(self.ts_threshold_fraction, "fitしきい値")

            ts = read_spe(thomson_file)
            ts_spectrum = spectrum_from_image(ts.image, y_min, y_max)
            ts_fit, ts_smooth, ts_baseline, ts_fit_curve = fit_broad_gaussian(
                ts_spectrum,
                fit_min=ts_fit_min,
                fit_max=ts_fit_max,
                baseline_left_min=ts_baseline_left_min,
                baseline_left_max=ts_baseline_left_max,
                baseline_right_min=ts_baseline_right_min,
                baseline_right_max=ts_baseline_right_max,
                median_width=ts_median_width,
                threshold_fraction=ts_threshold_fraction,
                fixed_center=center if self.ts_fix_center.get() else None,
            )

            if self.area_kind.get() == "gaussian":
                ts_area = ts_fit.gaussian_area
            elif self.area_kind.get() == "multi_stokes":
                ts_area = ts_fit.gaussian_area
            else:
                ts_area = ts_fit.direct_area

            self._positive(thomson_shots, "TS shot数")
            self._positive(thomson_energy_mj, "TS Energy")
            gas_density = gas_density_from_pressure(pressure, gas_temp)
            drift_note = "single"
            k_start = float("nan")
            k_end = float("nan")
            drift_rate = float("nan")

            if self.calibration_mode.get() == "drift":
                t_start = self._float(self.raman_start_time_h, "開始時刻")
                t_end = self._float(self.raman_end_time_h, "終了時刻")
                t_ts = self._float(self.thomson_time_h, "TS時刻")
                if t_end == t_start:
                    raise ValueError("開始時刻と終了時刻は異なる値にしてください。")

                start_pressure = self._float(self.raman_start_pressure_pa, "開始圧力")
                end_pressure = self._float(self.raman_end_pressure_pa, "終了圧力")
                start_shots = self._positive(self._float(self.raman_start_shots, "開始shot数"), "開始shot数")
                end_shots = self._positive(self._float(self.raman_end_shots, "終了shot数"), "終了shot数")
                start_energy = self._positive(self._float(self.raman_start_energy_mj, "開始Energy"), "開始Energy")
                end_energy = self._positive(self._float(self.raman_end_energy_mj, "終了Energy"), "終了Energy")

                raman_start_info = self._fit_raman_signal(
                    raman_start_file,
                    raman_center,
                    raman_stokes_min,
                    raman_stokes_max,
                    raman_max_peaks,
                    raman_peak_threshold,
                )
                raman_end_info = self._fit_raman_signal(
                    raman_end_file,
                    raman_center,
                    raman_stokes_min,
                    raman_stokes_max,
                    raman_max_peaks,
                    raman_peak_threshold,
                )
                start_gas_density = gas_density_from_pressure(start_pressure, gas_temp)
                end_gas_density = gas_density_from_pressure(end_pressure, gas_temp)
                k_start = throughput_from_raman(
                    raman_stokes_counts=float(raman_start_info["area"]) / start_energy,
                    gas_density_m3=start_gas_density,
                    raman_stokes_cross_section_m2=raman_dsigma,
                    raman_shots=start_shots,
                )
                k_end = throughput_from_raman(
                    raman_stokes_counts=float(raman_end_info["area"]) / end_energy,
                    gas_density_m3=end_gas_density,
                    raman_stokes_cross_section_m2=raman_dsigma,
                    raman_shots=end_shots,
                )
                fraction = (t_ts - t_start) / (t_end - t_start)
                throughput_k = k_start + (k_end - k_start) * fraction
                drift_rate = (k_end - k_start) / (t_end - t_start)
                raman_fit = raman_start_info["fit"]
                raman_area = float(raman_start_info["area"])
                raman_r_squared = float(raman_start_info["r_squared"])
                raman_peak_count = int(raman_start_info["peak_count"])
                raman_peak_pixels = str(raman_start_info["peak_pixels"])
                drift_note = f"drift_interpolation: t={t_ts:g} h, fraction={fraction:.4f}"
            else:
                self._positive(raman_shots, "ラマンshot数")
                self._positive(raman_energy_mj, "ラマンEnergy")
                raman_info = self._fit_raman_signal(
                    raman_file,
                    raman_center,
                    raman_stokes_min,
                    raman_stokes_max,
                    raman_max_peaks,
                    raman_peak_threshold,
                )
                raman_fit = raman_info["fit"]
                raman_area = float(raman_info["area"])
                raman_r_squared = float(raman_info["r_squared"])
                raman_peak_count = int(raman_info["peak_count"])
                raman_peak_pixels = str(raman_info["peak_pixels"])
                throughput_k = throughput_from_raman(
                    raman_stokes_counts=raman_area / raman_energy_mj,
                    gas_density_m3=gas_density,
                    raman_stokes_cross_section_m2=raman_dsigma,
                    raman_shots=raman_shots,
                )

            ne = electron_density_from_thomson_counts(
                thomson_counts=ts_area / thomson_energy_mj,
                throughput_k_m=throughput_k,
                thomson_cross_section_m2=thomson_dsigma,
                thomson_shots=thomson_shots,
                correction_factor=correction,
            )
            te = electron_temperature_from_width(ts_fit.sigma_pixel, nm_per_pixel, laser_nm, angle, inst_sigma)
            alpha = scattering_parameter_alpha(ne, te, laser_nm, angle)

            result_path = out_dir / "latest_lts_result.csv"
            ts_curve_path = out_dir / "latest_thomson_fit_curve.csv"
            write_fit_curve(ts_curve_path, ts_spectrum, ts_smooth, ts_baseline, ts_fit_curve)
            write_summary(
                result_path,
                {
                    "thomson_center": (ts_fit.center_pixel, "pixel"),
                    "thomson_sigma": (ts_fit.sigma_pixel, "pixel"),
                    "thomson_fwhm": (ts_fit.fwhm_pixel, "pixel"),
                    "thomson_area": (ts_area, "count_pixel"),
                    "thomson_r_squared": (ts_fit.r_squared, ""),
                    "thomson_fit_min": (ts_fit_min, "pixel"),
                    "thomson_fit_max": (ts_fit_max, "pixel"),
                    "thomson_baseline_left_min": (ts_baseline_left_min, "pixel"),
                    "thomson_baseline_left_max": (ts_baseline_left_max, "pixel"),
                    "thomson_baseline_right_min": (ts_baseline_right_min, "pixel"),
                    "thomson_baseline_right_max": (ts_baseline_right_max, "pixel"),
                    "thomson_median_width": (ts_median_width, "pixel"),
                    "thomson_threshold_fraction": (ts_threshold_fraction, ""),
                    "thomson_fixed_center": (int(self.ts_fix_center.get()), "0_or_1"),
                    "raman_center": (raman_fit.center_pixel, "pixel"),
                    "raman_sigma": (raman_fit.sigma_pixel, "pixel"),
                    "raman_area": (raman_area, "count_pixel"),
                    "raman_r_squared": (raman_r_squared, ""),
                    "raman_peak_count": (raman_peak_count, "peaks"),
                    "gas_density": (gas_density, "m^-3"),
                    "calibration_mode": (drift_note, ""),
                    "throughput_k": (throughput_k, "m/mJ"),
                    "throughput_k_start": (k_start, "m/mJ"),
                    "throughput_k_end": (k_end, "m/mJ"),
                    "throughput_k_drift_rate": (drift_rate, "m/mJ/h"),
                    "thomson_energy": (thomson_energy_mj, "mJ"),
                    "electron_density_m3": (ne, "m^-3"),
                    "electron_density_cm3": (ne / 1e6, "cm^-3"),
                    "electron_temperature": (te, "eV"),
                    "scattering_alpha": (alpha, ""),
                },
            )

            self._log(
                "\n".join(
                    [
                        "解析が完了しました。",
                        f"トムソン中心: {ts_fit.center_pixel:.3f} pixel",
                        f"トムソンFWHM: {ts_fit.fwhm_pixel:.3f} pixel",
                        f"トムソンR^2: {ts_fit.r_squared:.4f}",
                        f"ラマン中心: {raman_fit.center_pixel:.3f} pixel",
                        f"ラマンFWHM: {raman_fit.fwhm_pixel:.3f} pixel",
                        f"ラマンR^2: {raman_r_squared:.4f}",
                        f"ラマンピーク数: {raman_peak_count}",
                        f"ラマンピーク位置: {raman_peak_pixels} pixel",
                        f"n_gas: {gas_density:.6e} m^-3",
                        f"校正モード: {drift_note}",
                        f"k: {throughput_k:.6e} m/mJ",
                        f"k_start: {k_start:.6e} m/mJ",
                        f"k_end: {k_end:.6e} m/mJ",
                        f"ne: {ne:.6e} m^-3  ({ne / 1e6:.6e} cm^-3)",
                        f"Te: {te:.6e} eV",
                        f"alpha: {alpha:.4f}",
                        f"保存先: {result_path}",
                        f"TSフィット曲線・残差CSV: {ts_curve_path}",
                        "",
                    ]
                )
            )
        except Exception as exc:
            messagebox.showerror("解析エラー", str(exc))
            self._log(f"エラー: {exc}\n")

    def analyze_subtraction(self) -> None:
        try:
            if not self.subtract_save_csv.get() and not self.subtract_save_spe.get():
                raise ValueError("CSV保存またはSPE保存の少なくとも一方を選択してください。")

            out_dir = Path(self.subtract_out_dir.get())
            out_dir.mkdir(parents=True, exist_ok=True)
            scale = self._float(self.subtract_scale, "差し引き倍率")
            results = self._make_subtracted_images()
            saved_count = 0
            log_lines = [
                "差し引き保存が完了しました。",
                f"差し引くSPE数: {len(self.subtract_background_files)}",
                f"差し引き倍率: {scale:g}",
                f"保存先: {out_dir}",
            ]

            for target_path, result_image in results:
                stem = target_path.stem

                if self.subtract_save_csv.get():
                    csv_path = out_dir / f"{stem}_subtracted.csv"
                    save_image_csv(csv_path, result_image)
                    saved_count += 1
                    log_lines.append(f"CSV: {csv_path.name}")

                if self.subtract_save_spe.get():
                    spe_path = out_dir / f"{stem}_subtracted.spe"
                    save_spe_like(spe_path, result_image, target_path)
                    saved_count += 1
                    log_lines.append(f"SPE: {spe_path.name}")

            first_path, first_image = results[0]
            self._draw_image_canvas(
                self.subtract_image_canvas,
                first_image,
                f"{first_path.name} 差し引き後画像",
                "subtract_image_photo",
            )
            self._draw_spectrum_canvas(
                self.subtract_spectrum_canvas,
                first_image.sum(axis=0),
                "差し引き後スペクトル",
            )
            self.subtract_log.insert("end", "\n".join(log_lines) + f"\n保存ファイル数: {saved_count}\n\n")
            self.subtract_log.see("end")
        except Exception as exc:
            messagebox.showerror("差し引きエラー", str(exc))
            self.subtract_log.insert("end", f"エラー: {exc}\n")
            self.subtract_log.see("end")

    def preview_subtraction(self) -> None:
        try:
            results = self._make_subtracted_images(preview_only=True)
            target_path, result_image = results[0]
            self._draw_image_canvas(
                self.subtract_image_canvas,
                result_image,
                f"{target_path.name} 差し引き後画像",
                "subtract_image_photo",
            )
            self._draw_spectrum_canvas(
                self.subtract_spectrum_canvas,
                result_image.sum(axis=0),
                "差し引き後スペクトル",
            )
            self.subtract_log.insert("end", f"プレビュー更新: {target_path.name}\n")
            self.subtract_log.see("end")
        except Exception as exc:
            messagebox.showerror("プレビューエラー", str(exc))
            self.subtract_log.insert("end", f"エラー: {exc}\n")
            self.subtract_log.see("end")

    def fit_subtracted_spectrum(self) -> None:
        try:
            target_path, image = self._fit_image_for_subtraction_tab()
            y_min = self._optional_int(self.y_min, "TS y最小")
            y_max = self._optional_int(self.y_max, "TS y最大")
            spectrum = spectrum_from_image(image, y_min, y_max)
            x_all = np.arange(spectrum.size, dtype=np.float64)

            if self.subtract_fit_kind.get() == "thomson":
                center = self._float(self.fixed_center, "レーザー中心")
                ts_fit_min = self._int(self.ts_fit_min, "TS fit最小")
                ts_fit_max = self._int(self.ts_fit_max, "TS fit最大")
                ts_baseline_left_min = self._int(self.ts_baseline_left_min, "背景左最小")
                ts_baseline_left_max = self._int(self.ts_baseline_left_max, "背景左最大")
                ts_baseline_right_min = self._int(self.ts_baseline_right_min, "背景右最小")
                ts_baseline_right_max = self._int(self.ts_baseline_right_max, "背景右最大")
                ts_median_width = self._int(self.ts_median_width, "平滑化幅")
                ts_threshold_fraction = self._float(self.ts_threshold_fraction, "fitしきい値")
                fit, smooth, _baseline, fit_curve = fit_broad_gaussian(
                    spectrum,
                    fit_min=ts_fit_min,
                    fit_max=ts_fit_max,
                    baseline_left_min=ts_baseline_left_min,
                    baseline_left_max=ts_baseline_left_max,
                    baseline_right_min=ts_baseline_right_min,
                    baseline_right_max=ts_baseline_right_max,
                    median_width=ts_median_width,
                    threshold_fraction=ts_threshold_fraction,
                    fixed_center=center if self.ts_fix_center.get() else None,
                )
                residual = smooth - fit_curve
                self._draw_fit_overlay_canvas(
                    self.subtract_fit_canvas,
                    x_all,
                    smooth,
                    fit_curve,
                    f"トムソンフィット: {target_path.name}",
                )
                self._draw_spectrum_canvas(self.subtract_residual_canvas, residual, "残差", x_start=0)
                self.subtract_log.insert(
                    "end",
                    "\n".join(
                        [
                            f"トムソンフィット完了: {target_path.name}",
                            f"center={fit.center_pixel:.3f} pixel",
                            f"FWHM={fit.fwhm_pixel:.3f} pixel",
                            f"area={fit.gaussian_area:.6e} count*pixel",
                            f"R^2={fit.r_squared:.4f}",
                            "",
                        ]
                    ),
                )
            else:
                raman_center = self._optional_float(self.raman_center, "ラマン中心")
                fit = fit_gaussian_log_parabola(
                    spectrum,
                    peak_pixel=None if raman_center is None else int(round(raman_center)),
                    window=45,
                    sideband=100,
                    fixed_center=raman_center,
                )
                fit_curve = fit.baseline + fit.amplitude * np.exp(-0.5 * ((x_all - fit.center_pixel) / fit.sigma_pixel) ** 2)
                residual = spectrum.astype(np.float64) - fit_curve
                self._draw_fit_overlay_canvas(
                    self.subtract_fit_canvas,
                    x_all,
                    spectrum.astype(np.float64),
                    fit_curve,
                    f"ラマンフィット: {target_path.name}",
                )
                self._draw_spectrum_canvas(self.subtract_residual_canvas, residual, "残差", x_start=0)
                self.subtract_log.insert(
                    "end",
                    "\n".join(
                        [
                            f"ラマンフィット完了: {target_path.name}",
                            f"center={fit.center_pixel:.3f} pixel",
                            f"FWHM={fit.fwhm_pixel:.3f} pixel",
                            f"area={fit.gaussian_area:.6e} count*pixel",
                            f"R^2={fit.r_squared:.4f}",
                            "",
                        ]
                    ),
                )
            self.subtract_log.see("end")
        except Exception as exc:
            messagebox.showerror("フィットエラー", str(exc))
            self.subtract_log.insert("end", f"エラー: {exc}\n")
            self.subtract_log.see("end")

    def show_spe_viewer(self) -> None:
        try:
            path = Path(self.viewer_path.get())
            if not path.exists():
                raise FileNotFoundError(f"SPEファイルが見つかりません: {path}")
            spe = read_spe(path)
            x0, x1, y0, y1 = self._viewer_roi(spe.image.shape)
            spectrum = spe.image[y0:y1, x0:x1].sum(axis=0)

            self._draw_image_canvas(self.viewer_image_canvas, spe.image, path.name, "viewer_image_photo", roi=(x0, x1, y0, y1))
            self._draw_spectrum_canvas(
                self.viewer_spectrum_canvas,
                spectrum,
                f"ROI積算スペクトル x={x0}:{x1}, y={y0}:{y1}",
                x_start=x0,
            )
            self.viewer_log.delete("1.0", "end")
            self.viewer_log.insert(
                "end",
                "\n".join(
                    [
                        "表示しました。",
                        f"ファイル: {path}",
                        f"画像サイズ: {spe.xdim} x {spe.ydim} pixel",
                        f"frames: {spe.frames}",
                        f"dtype_code: {spe.dtype_code}",
                        f"表示範囲: x={x0}:{x1} pixel, y={y0}:{y1} pixel",
                    ]
                )
                + "\n",
            )
        except Exception as exc:
            messagebox.showerror("ビューアエラー", str(exc))
            self.viewer_log.insert("end", f"エラー: {exc}\n")

    def _log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _browse_pressure_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.pressure_folder.set(path)
            self._load_pressure_folder()

    def _clear_pressure_rows(self) -> None:
        for child in self.pressure_table.winfo_children():
            child.destroy()
        self.pressure_rows.clear()
        self.pressure_result.delete("1.0", "end")
        self.pressure_plot.delete("all")

    def _load_pressure_folder(self) -> None:
        folder = Path(self.pressure_folder.get())
        if not folder.exists():
            messagebox.showerror("フォルダエラー", f"フォルダが見つかりません: {folder}")
            return
        self._clear_pressure_rows()
        files = sorted(folder.glob("*.spe"))
        if not files:
            messagebox.showinfo("SPEなし", "選択フォルダに .spe ファイルがありません。")
            return
        self._populate_pressure_rows(files)

    def analyze_pressure_dependence(self) -> None:
        try:
            if not self.pressure_rows:
                raise ValueError("先にSPEフォルダを読み込んでください。")

            center = self._optional_float(self.pressure_peak_center, "ラマン中心")
            results = []

            for path, pressure_var in self.pressure_rows:
                pressure_text = pressure_var.get().strip()
                if not pressure_text:
                    continue
                pressure_pa = float(pressure_text)
                spe = read_spe(path)
                spectrum = spectrum_from_image(spe.image, None, None)
                fit = fit_gaussian_log_parabola(
                    spectrum,
                    peak_pixel=None if center is None else int(round(center)),
                    window=45,
                    sideband=100,
                    fixed_center=center,
                )
                if self.pressure_signal_kind.get() == "area":
                    intensity = fit.gaussian_area
                    unit = "count*pixel"
                else:
                    intensity = fit.amplitude
                    unit = "counts"
                results.append(
                    {
                        "file": path.name,
                        "pressure_pa": pressure_pa,
                        "intensity": intensity,
                        "unit": unit,
                        "center": fit.center_pixel,
                        "fwhm": fit.fwhm_pixel,
                        "r_squared": fit.r_squared,
                    }
                )

            if len(results) < 2:
                raise ValueError("グラフ化には、圧力を入力したデータが2点以上必要です。")

            results.sort(key=lambda item: item["pressure_pa"])
            folder = Path(self.pressure_folder.get())
            csv_path = folder / "raman_pressure_dependence.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["file", "pressure_pa", "intensity", "unit", "center", "fwhm", "r_squared"],
                )
                writer.writeheader()
                writer.writerows(results)

            self._draw_pressure_plot(results)
            self._show_pressure_results(results, csv_path)
        except Exception as exc:
            messagebox.showerror("圧力依存性解析エラー", str(exc))
            self.pressure_result.insert("end", f"エラー: {exc}\n")

    def _draw_pressure_plot(self, results: list[dict[str, float | str]]) -> None:
        canvas = self.pressure_plot
        canvas.delete("all")
        width = max(canvas.winfo_width(), 640)
        height = max(canvas.winfo_height(), 360)
        left, right, top, bottom = 80, 30, 30, 70
        plot_w = width - left - right
        plot_h = height - top - bottom

        xs = [float(item["pressure_pa"]) for item in results]
        ys = [float(item["intensity"]) for item in results]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.08
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        def px(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def py(y: float) -> float:
            return top + plot_h - (y - y_min) / (y_max - y_min) * plot_h

        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="black")
        canvas.create_line(left, top, left, top + plot_h, fill="black")
        canvas.create_text(left + plot_w / 2, height - 25, text="圧力 [Pa]")
        y_label = "ピーク高さ [counts]" if self.pressure_signal_kind.get() == "amplitude" else "ガウス面積 [count*pixel]"
        canvas.create_text(22, top + plot_h / 2, text=y_label, angle=90)

        for i in range(6):
            x_val = x_min + (x_max - x_min) * i / 5
            x_pos = px(x_val)
            canvas.create_line(x_pos, top + plot_h, x_pos, top + plot_h + 4, fill="black")
            canvas.create_text(x_pos, top + plot_h + 18, text=f"{x_val:.2g}")
            y_val = y_min + (y_max - y_min) * i / 5
            y_pos = py(y_val)
            canvas.create_line(left - 4, y_pos, left, y_pos, fill="black")
            canvas.create_text(left - 8, y_pos, text=f"{y_val:.2g}", anchor="e")

        points = [(px(x), py(y)) for x, y in zip(xs, ys)]
        for i in range(len(points) - 1):
            canvas.create_line(*points[i], *points[i + 1], fill="#1f5fbf", width=2)
        for x_pos, y_pos in points:
            canvas.create_oval(x_pos - 4, y_pos - 4, x_pos + 4, y_pos + 4, fill="#d34a35", outline="")

    def _show_pressure_results(self, results: list[dict[str, float | str]], csv_path: Path) -> None:
        self.pressure_result.delete("1.0", "end")
        self.pressure_result.insert("end", "解析が完了しました。\n")
        self.pressure_result.insert("end", f"保存先: {csv_path}\n")
        self.pressure_result.insert("end", "圧力 [Pa], 強度, R^2\n")
        for item in results:
            self.pressure_result.insert(
                "end",
                f"{float(item['pressure_pa']):.6g}, {float(item['intensity']):.6e} {item['unit']}, {float(item['r_squared']):.4f}\n",
            )


def main() -> None:
    app = LtsAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
