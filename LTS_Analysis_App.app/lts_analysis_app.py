from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_density import (  # noqa: E402
    PAPER_N2_STOKES_CROSS_SECTION_M2,
    PAPER_THOMSON_CROSS_SECTION_M2,
    electron_density_from_paper_raman_calibration,
    electron_temperature_from_width,
    gas_density_from_pressure,
    scattering_parameter_alpha,
)
from fit_peak import fit_gaussian_log_parabola  # noqa: E402
from fit_thomson import fit_broad_gaussian  # noqa: E402
from read_spe import read_spe, spectrum_from_image  # noqa: E402


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
        self.geometry("980x720")
        self.minsize(900, 640)

        self.thomson_path = tk.StringVar()
        default_raman = APP_DIR / "virtual_raman.spe"
        self.raman_path = tk.StringVar(value=str(default_raman) if default_raman.exists() else "")
        self.out_dir = tk.StringVar(value=str(APP_DIR))

        self.y_min = tk.StringVar(value="600")
        self.y_max = tk.StringVar(value="970")
        self.fixed_center = tk.StringVar(value="536.8749")
        self.raman_center = tk.StringVar(value="650")
        self.raman_mask_min = tk.StringVar(value="")
        self.raman_mask_max = tk.StringVar(value="")
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
        self.correction_factor = tk.StringVar(value="1")
        self.area_kind = tk.StringVar(value="gaussian")

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        root = ttk.Frame(notebook, padding=12)
        notebook.add(root, text="ラマン校正TS解析")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        file_box = ttk.LabelFrame(root, text="ファイル", padding=10)
        file_box.grid(row=0, column=0, sticky="ew")
        file_box.columnconfigure(1, weight=1)
        self._file_row(file_box, 0, "トムソンSPE", self.thomson_path)
        self._file_row(file_box, 1, "ラマンSPE", self.raman_path)
        self._dir_row(file_box, 2, "保存先", self.out_dir)

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
        self._entry(param_box, 2, 4, "補正係数", self.correction_factor, "-")

        area_frame = ttk.Frame(param_box)
        area_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(area_frame, text="面積").pack(side="left")
        ttk.Radiobutton(area_frame, text="ガウス面積", value="gaussian", variable=self.area_kind).pack(side="left", padx=8)
        ttk.Radiobutton(area_frame, text="直接積分", value="direct", variable=self.area_kind).pack(side="left")

        actions = ttk.Frame(root)
        actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(actions, text="解析", command=self.analyze).pack(side="left")
        ttk.Button(actions, text="ログ消去", command=lambda: self.log.delete("1.0", "end")).pack(side="left", padx=8)

        self.log = tk.Text(root, height=20, wrap="word")
        self.log.grid(row=3, column=0, sticky="nsew", pady=(10, 0))

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
        ttk.Label(value_frame, text=unit, width=8).pack(side="left", padx=(4, 0))

    def _browse_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("SPE files", "*.spe"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _browse_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _float(self, var: tk.StringVar, name: str) -> float:
        try:
            return float(var.get())
        except ValueError as exc:
            raise ValueError(f"{name} must be a number: {var.get()}") from exc

    def _int(self, var: tk.StringVar, name: str) -> int:
        try:
            return int(float(var.get()))
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer: {var.get()}") from exc

    def _optional_float(self, var: tk.StringVar, name: str) -> float | None:
        text = var.get().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number or blank: {var.get()}") from exc

    def _optional_int(self, var: tk.StringVar, name: str) -> int | None:
        value = self._optional_float(var, name)
        if value is None:
            return None
        return int(round(value))

    def analyze(self) -> None:
        try:
            out_dir = Path(self.out_dir.get())
            out_dir.mkdir(parents=True, exist_ok=True)

            thomson_file = Path(self.thomson_path.get())
            raman_file = Path(self.raman_path.get())
            if not thomson_file.exists():
                raise FileNotFoundError(f"トムソンSPEが見つかりません: {thomson_file}")
            if not raman_file.exists():
                raise FileNotFoundError(f"ラマンSPEが見つかりません: {raman_file}")

            y_min = self._int(self.y_min, "TS y min")
            y_max = self._int(self.y_max, "TS y max")
            center = self._float(self.fixed_center, "Center pixel")
            raman_center = self._optional_float(self.raman_center, "Raman center")
            raman_mask_min = self._optional_int(self.raman_mask_min, "Stop min px")
            raman_mask_max = self._optional_int(self.raman_mask_max, "Stop max px")
            nm_per_pixel = self._float(self.nm_per_pixel, "nm/pixel")
            laser_nm = self._float(self.laser_wavelength_nm, "Laser nm")
            angle = self._float(self.scattering_angle_deg, "Angle deg")
            inst_sigma = self._float(self.instrument_sigma_pixel, "Inst sigma px")
            pressure = self._float(self.pressure_pa, "Pressure Pa")
            gas_temp = self._float(self.gas_temperature_k, "Gas K")
            raman_dsigma = self._float(self.raman_dsigma, "Raman dσ")
            thomson_dsigma = self._float(self.thomson_dsigma, "Thomson dσ")
            raman_shots = self._float(self.raman_shots, "Raman shots")
            thomson_shots = self._float(self.thomson_shots, "Thomson shots")
            correction = self._float(self.correction_factor, "Correction")

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

            raman = read_spe(raman_file)
            raman_spectrum = spectrum_from_image(raman.image, None, None)
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
                ts_area = ts_fit.gaussian_area
                raman_area = raman_fit.gaussian_area
            else:
                ts_area = ts_fit.direct_area
                raman_area = raman_fit.direct_area
                if raman_mask_min is not None and raman_mask_max is not None:
                    self._log("警告: ラマンピークがストップで欠けている場合、直接積分面積は推奨しません。ガウス面積を使ってください。")

            gas_density = gas_density_from_pressure(pressure, gas_temp)
            ne, throughput_k = electron_density_from_paper_raman_calibration(
                thomson_counts=ts_area,
                raman_stokes_counts=raman_area,
                gas_density_m3=gas_density,
                raman_stokes_cross_section_m2=raman_dsigma,
                thomson_cross_section_m2=thomson_dsigma,
                thomson_shots=thomson_shots,
                raman_shots=raman_shots,
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
                    "raman_r_squared": (raman_fit.r_squared, ""),
                    "gas_density": (gas_density, "m^-3"),
                    "throughput_k": (throughput_k, "m"),
                    "raman_effective_cross_section": (raman_dsigma, "m^2"),
                    "thomson_cross_section": (thomson_dsigma, "m^2"),
                    "raman_shots": (raman_shots, "shots"),
                    "thomson_shots": (thomson_shots, "shots"),
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
                        f"ラマンR^2: {raman_fit.r_squared:.4f}",
                        f"面積種別: {self.area_kind.get()}",
                        f"n_gas: {gas_density:.6e} m^-3",
                        f"k: {throughput_k:.6e} m",
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

    def _log(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")


def main() -> None:
    app = LtsAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
