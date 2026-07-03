from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SPE_HEADER_BYTES = 4100


@dataclass
class SpeData:
    path: Path
    xdim: int
    ydim: int
    frames: int
    dtype_code: int
    image: np.ndarray


def _spe_dtype(dtype_code: int) -> np.dtype:
    # Common Princeton Instruments / LightField SPE datatype codes.
    mapping = {
        0: np.float32,
        1: np.int32,
        2: np.int16,
        3: np.uint16,
        8: np.uint32,
    }
    if dtype_code not in mapping:
        raise ValueError(f"Unsupported SPE datatype code: {dtype_code}")
    return np.dtype(mapping[dtype_code])


def read_spe(path: str | Path) -> SpeData:
    path = Path(path)
    raw = path.read_bytes()

    xdim = int(np.frombuffer(raw, dtype="<u2", count=1, offset=42)[0])
    ydim = int(np.frombuffer(raw, dtype="<u2", count=1, offset=656)[0])
    frames = int(np.frombuffer(raw, dtype="<i4", count=1, offset=1446)[0])
    dtype_code = int(np.frombuffer(raw, dtype="<u2", count=1, offset=108)[0])
    dtype = _spe_dtype(dtype_code)

    count = xdim * ydim * max(frames, 1)
    data = np.frombuffer(raw, dtype=dtype.newbyteorder("<"), count=count, offset=SPE_HEADER_BYTES)

    if frames <= 1:
        image = data.reshape((ydim, xdim))
    else:
        image = data.reshape((frames, ydim, xdim))[0]

    return SpeData(path=path, xdim=xdim, ydim=ydim, frames=frames, dtype_code=dtype_code, image=image)


def spectrum_from_image(image: np.ndarray, y_min: int | None, y_max: int | None) -> np.ndarray:
    y0 = 0 if y_min is None else y_min
    y1 = image.shape[0] if y_max is None else y_max
    if not (0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"Invalid y range: {y0}:{y1} for image height {image.shape[0]}")
    return image[y0:y1, :].sum(axis=0)


def save_spectrum_csv(path: Path, spectrum: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel", "integrated_counts"])
        for pixel, value in enumerate(spectrum):
            writer.writerow([pixel, float(value)])


def save_image_csv(path: Path, image: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y/x", *range(image.shape[1])])
        for y, row in enumerate(image):
            writer.writerow([y, *[float(value) for value in row]])


def save_spe_like(path: Path, image: np.ndarray, reference_path: str | Path) -> None:
    """Save a simple SPE file by reusing the reference header and writing float32 image data.

    The original SPE header keeps most acquisition metadata. The datatype is changed to
    float32 because subtraction can produce negative and fractional values.
    """
    reference_path = Path(reference_path)
    raw = bytearray(reference_path.read_bytes())
    if len(raw) < SPE_HEADER_BYTES:
        raise ValueError(f"SPEヘッダが短すぎます: {reference_path}")

    ydim, xdim = image.shape
    raw[42:44] = np.array([xdim], dtype="<u2").tobytes()
    raw[656:658] = np.array([ydim], dtype="<u2").tobytes()
    raw[108:110] = np.array([0], dtype="<u2").tobytes()
    raw[1446:1450] = np.array([1], dtype="<i4").tobytes()

    header = bytes(raw[:SPE_HEADER_BYTES])
    data = np.asarray(image, dtype="<f4").tobytes()
    path.write_bytes(header + data)


def save_preview_png(path: Path, image: np.ndarray) -> None:
    arr = image.astype(np.float64)
    lo, hi = np.percentile(arr, [1, 99.7])
    if hi <= lo:
        hi = arr.max() if arr.max() > lo else lo + 1
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((scaled * 255).astype(np.uint8))
    img.save(path)


def save_spectrum_png(path: Path, spectrum: np.ndarray) -> None:
    width, height = 1200, 650
    margin_left, margin_right = 75, 25
    margin_top, margin_bottom = 25, 65
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    arr = spectrum.astype(np.float64)
    y_min = float(np.percentile(arr, 1))
    y_max = float(np.percentile(arr, 99.8))
    if y_max <= y_min:
        y_max = float(arr.max() if arr.max() > y_min else y_min + 1)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[margin_top : margin_top + plot_h, margin_left : margin_left + plot_w] = 248

    # Axes
    canvas[margin_top + plot_h, margin_left : margin_left + plot_w + 1] = 40
    canvas[margin_top : margin_top + plot_h + 1, margin_left] = 40

    xs = np.linspace(margin_left, margin_left + plot_w - 1, arr.size).astype(int)
    ys = margin_top + plot_h - 1 - np.clip((arr - y_min) / (y_max - y_min), 0, 1) * (plot_h - 1)
    ys = ys.astype(int)

    for i in range(len(xs) - 1):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        line_x = np.linspace(x0, x1, steps + 1).astype(int)
        line_y = np.linspace(y0, y1, steps + 1).astype(int)
        canvas[line_y, line_x] = (20, 90, 180)

    Image.fromarray(canvas).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a LightField/Princeton SPE file and export a 1D spectrum.")
    parser.add_argument("spe_file", help="Path to .spe file")
    parser.add_argument("--out-dir", default=".", help="Directory for CSV and PNG outputs")
    parser.add_argument("--y-min", type=int, default=None, help="First detector row to integrate")
    parser.add_argument("--y-max", type=int, default=None, help="One past last detector row to integrate")
    args = parser.parse_args()

    spe = read_spe(args.spe_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = spe.path.stem
    csv_path = out_dir / f"{stem}_spectrum.csv"
    png_path = out_dir / f"{stem}_preview.png"
    spectrum_png_path = out_dir / f"{stem}_spectrum.png"

    spectrum = spectrum_from_image(spe.image, args.y_min, args.y_max)
    save_spectrum_csv(csv_path, spectrum)
    save_preview_png(png_path, spe.image)
    save_spectrum_png(spectrum_png_path, spectrum)

    print(f"file: {spe.path}")
    print(f"size: {spe.xdim} x {spe.ydim}, frames={spe.frames}, dtype_code={spe.dtype_code}")
    print(f"spectrum CSV: {csv_path}")
    print(f"preview PNG: {png_path}")
    print(f"spectrum PNG: {spectrum_png_path}")
    print(f"integrated rows: {0 if args.y_min is None else args.y_min}:{spe.ydim if args.y_max is None else args.y_max}")


if __name__ == "__main__":
    main()
