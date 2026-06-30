from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SPE_HEADER_BYTES = 4100


def make_virtual_raman_image(
    xdim: int = 1024,
    ydim: int = 1024,
    center_pixel: float = 650.0,
    sigma_pixel: float = 4.0,
    y_center: float = 520.0,
    y_sigma: float = 130.0,
    amplitude: float = 5200.0,
    baseline: float = 450.0,
    noise_sigma: float = 18.0,
    seed: int = 7,
    stop_min_pixel: int | None = None,
    stop_max_pixel: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.arange(xdim, dtype=np.float64)
    y = np.arange(ydim, dtype=np.float64)
    spectral = np.exp(-0.5 * ((x - center_pixel) / sigma_pixel) ** 2)
    slit = np.exp(-0.5 * ((y - y_center) / y_sigma) ** 2)
    image = baseline + amplitude * np.outer(slit, spectral)
    if stop_min_pixel is not None and stop_max_pixel is not None:
        stop_min = max(0, int(stop_min_pixel))
        stop_max = min(xdim - 1, int(stop_max_pixel))
        if stop_min <= stop_max:
            image[:, stop_min : stop_max + 1] = baseline
    image += rng.normal(0.0, noise_sigma, size=image.shape)
    image = np.clip(image, 0, np.iinfo(np.uint16).max)
    return image.astype("<u2")


def write_spe(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ydim, xdim = image.shape
    header = bytearray(SPE_HEADER_BYTES)
    header[42:44] = int(xdim).to_bytes(2, "little", signed=False)
    header[656:658] = int(ydim).to_bytes(2, "little", signed=False)
    header[108:110] = int(3).to_bytes(2, "little", signed=False)  # uint16
    header[1446:1450] = int(1).to_bytes(4, "little", signed=True)

    xml = b"<SpeFormat><VirtualData kind=\"raman\" /></SpeFormat>"
    xml_offset = SPE_HEADER_BYTES + image.nbytes
    header[678:686] = int(xml_offset).to_bytes(8, "little", signed=True)
    path.write_bytes(bytes(header) + image.tobytes(order="C") + xml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a virtual Raman SPE file for pipeline testing.")
    parser.add_argument("--out", required=True, help="Output .spe path")
    parser.add_argument("--center-pixel", type=float, default=650.0)
    parser.add_argument("--sigma-pixel", type=float, default=4.0)
    parser.add_argument("--amplitude", type=float, default=5200.0)
    parser.add_argument("--baseline", type=float, default=450.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stop-min-pixel", type=int, default=None)
    parser.add_argument("--stop-max-pixel", type=int, default=None)
    args = parser.parse_args()

    image = make_virtual_raman_image(
        center_pixel=args.center_pixel,
        sigma_pixel=args.sigma_pixel,
        amplitude=args.amplitude,
        baseline=args.baseline,
        seed=args.seed,
        stop_min_pixel=args.stop_min_pixel,
        stop_max_pixel=args.stop_max_pixel,
    )
    write_spe(args.out, image)
    print(f"virtual Raman SPE: {Path(args.out)}")
    print(f"size: {image.shape[1]} x {image.shape[0]}, dtype=uint16")
    print(f"center_pixel: {args.center_pixel}")
    print(f"sigma_pixel: {args.sigma_pixel}")
    if args.stop_min_pixel is not None and args.stop_max_pixel is not None:
        print(f"blocked pixels: {args.stop_min_pixel}:{args.stop_max_pixel}")


if __name__ == "__main__":
    main()
