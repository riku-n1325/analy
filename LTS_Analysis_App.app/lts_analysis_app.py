from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

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


def write_summary(path: Path, values: dict[str, tuple[float, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "unit"])
        for name, (value, unit) in values.items():
            writer.writerow([name, value, unit])


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
        self.raman_mask_min = tk.StringVar(value="")
        self.raman_mask_max = tk.StringVar(value="")
        self.raman_stokes_min = tk.StringVar(value="")
        self.raman_stokes_max = tk.StringVar(value="")
        self.raman_max_peaks = tk.StringVar(value="12")
        self.raman_peak_threshold = tk.StringVar(value="0.08")
        self.nm_per_pixel = tk.StringVar(value="0.021")
        self.laser_wavelength_nm = tk.StringVar(value="532")
        self.scattering_angle_deg = tk.StringVar(value="90")
        self.instrument_sigma_pixel = tk.StringVar(value="0")

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
        self.pressure_mask_min = tk.StringVar(value="")
        self.pressure_mask_max = tk.StringVar(value="")
        self.pressure_signal_kind = tk.StringVar(value="amplitude")
        self.pressure_rows: list[tuple[Path, tk.StringVar]] = []

        self.subtract_out_dir = tk.StringVar(value=str(APP_DIR))
        self.subtract_scale = tk.StringVar(value="1")
        self.subtract_save_csv = tk.BooleanVar(value=True)
        self.subtract_save_spe = tk.BooleanVar(value=True)
        self.subtract_target_files: list[Path] = []
        self.subtract_background_files: list[Path] = []

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self._build_calibration_tab(notebook)
        self._build_pressure_tab(notebook)
        self._build_subtraction_tab(notebook)

    def _build_calibration_tab(self, notebook: ttk.Notebook) -> None:
        root = ttk.Frame(notebook, padding=12)
        notebook.add(root, text="ラマン校正TS解析")
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
        self._entry(param_box, 0, 4, "ストップ最小", self.raman_mask_min, "pixel")
        self._entry(param_box, 0, 5, "ストップ最大", self.raman_mask_max, "pixel")

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
        self._entry(option_box, 0, 1, "ストップ最小", self.pressure_mask_min, "pixel")
        self._entry(option_box, 0, 2, "ストップ最大", self.pressure_mask_max, "pixel")

        signal_frame = ttk.Frame(option_box)
        signal_frame.grid(row=0, column=3, columnspan=2, sticky="w", padx=4)
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

        bottom = ttk.Frame(root)
        bottom.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(bottom, text="差し引き保存", command=self.analyze_subtraction).pack(side="left")
        ttk.Button(bottom, text="ログ消去", command=lambda: self.subtract_log.delete("1.0", "end")).pack(side="left", padx=8)

        self.subtract_log = tk.Text(root, height=10, wrap="word")
        self.subtract_log.grid(row=3, column=0, sticky="ew", pady=(10, 0))

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

    def _fit_raman_signal(
        self,
        raman_file: Path,
        raman_center: float | None,
        raman_mask_min: int | None,
        raman_mask_max: int | None,
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
                mask_min=raman_mask_min,
                mask_max=raman_mask_max,
                fixed_center=raman_center,
            )
            if self.area_kind.get() == "gaussian":
                area = raman_fit.gaussian_area
            else:
                area = raman_fit.direct_area
                if raman_mask_min is not None and raman_mask_max is not None:
                    self._log("警告: ラマンピークがストップで欠けている場合、直接積分面積は推奨しません。")
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
            raman_mask_min = self._optional_int(self.raman_mask_min, "ストップ最小")
            raman_mask_max = self._optional_int(self.raman_mask_max, "ストップ最大")
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

            ts = read_spe(thomson_file)
            ts_spectrum = spectrum_from_image(ts.image, y_min, y_max)
            ts_fit, _, _, _ = fit_broad_gaussian(
                ts_spectrum,
                fit_min=360,
                fit_max=760,
                baseline_left_min=180,
                baseline_left_max=330,
                baseline_right_min=800,
                baseline_right_max=1000,
                fixed_center=center,
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
                    raman_mask_min,
                    raman_mask_max,
                    raman_stokes_min,
                    raman_stokes_max,
                    raman_max_peaks,
                    raman_peak_threshold,
                )
                raman_end_info = self._fit_raman_signal(
                    raman_end_file,
                    raman_center,
                    raman_mask_min,
                    raman_mask_max,
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
                    raman_mask_min,
                    raman_mask_max,
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
            write_summary(
                result_path,
                {
                    "thomson_center": (ts_fit.center_pixel, "pixel"),
                    "thomson_sigma": (ts_fit.sigma_pixel, "pixel"),
                    "thomson_fwhm": (ts_fit.fwhm_pixel, "pixel"),
                    "thomson_area": (ts_area, "count_pixel"),
                    "thomson_r_squared": (ts_fit.r_squared, ""),
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
                        "",
                    ]
                )
            )
        except Exception as exc:
            messagebox.showerror("解析エラー", str(exc))
            self._log(f"エラー: {exc}\n")

    def analyze_subtraction(self) -> None:
        try:
            if not self.subtract_target_files:
                raise ValueError("差し引き対象SPEを1つ以上追加してください。")
            if not self.subtract_background_files:
                raise ValueError("差し引くSPEを1つ以上追加してください。")
            if not self.subtract_save_csv.get() and not self.subtract_save_spe.get():
                raise ValueError("CSV保存またはSPE保存の少なくとも一方を選択してください。")

            out_dir = Path(self.subtract_out_dir.get())
            out_dir.mkdir(parents=True, exist_ok=True)
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
            saved_count = 0
            log_lines = [
                "差し引き保存が完了しました。",
                f"差し引くSPE数: {len(self.subtract_background_files)}",
                f"差し引き倍率: {scale:g}",
                f"保存先: {out_dir}",
            ]

            for target_path in self.subtract_target_files:
                target = read_spe(target_path)
                target_image = target.image.astype(np.float64)
                if target_image.shape != background_mean.shape:
                    raise ValueError(
                        f"対象SPEと差し引くSPEの画像サイズが一致しません: {target_path.name} "
                        f"{target_image.shape} != {background_mean.shape}"
                    )

                result_image = target_image - scale * background_mean
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

            self.subtract_log.insert("end", "\n".join(log_lines) + f"\n保存ファイル数: {saved_count}\n\n")
            self.subtract_log.see("end")
        except Exception as exc:
            messagebox.showerror("差し引きエラー", str(exc))
            self.subtract_log.insert("end", f"エラー: {exc}\n")
            self.subtract_log.see("end")

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

        ttk.Label(self.pressure_table, text="ファイル名", width=36).grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Label(self.pressure_table, text="圧力", width=12).grid(row=0, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(self.pressure_table, text="単位", width=8).grid(row=0, column=2, sticky="w", padx=2, pady=2)

        for row, path in enumerate(files, start=1):
            pressure_var = tk.StringVar()
            ttk.Label(self.pressure_table, text=path.name, width=42).grid(row=row, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(self.pressure_table, textvariable=pressure_var, width=14).grid(row=row, column=1, sticky="w", padx=2, pady=2)
            ttk.Label(self.pressure_table, text="Pa", width=8).grid(row=row, column=2, sticky="w", padx=2, pady=2)
            self.pressure_rows.append((path, pressure_var))

    def analyze_pressure_dependence(self) -> None:
        try:
            if not self.pressure_rows:
                raise ValueError("先にSPEフォルダを読み込んでください。")

            center = self._optional_float(self.pressure_peak_center, "ラマン中心")
            mask_min = self._optional_int(self.pressure_mask_min, "ストップ最小")
            mask_max = self._optional_int(self.pressure_mask_max, "ストップ最大")
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
                    mask_min=mask_min,
                    mask_max=mask_max,
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
