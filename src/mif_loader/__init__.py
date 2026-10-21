from __future__ import annotations

import logging
import os
import re
import typing
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)

CANONICAL_AXES = "TCZYX"

# dtypes cv2.warpAffine can handle natively
_CV2_DTYPES = {
    np.dtype(t) for t in (np.uint8, np.uint16, np.int16, np.float32, np.float64)
}

# "key[N] = value;"  (value may contain spaces; trailing whitespace allowed)
_LINE_RE = re.compile(r"^\s*([A-Za-z]+)(\d+)?\s*=\s*(.+?)\s*;\s*$")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def rigid(sx: float, sy: float, rot: float, tx: float, ty: float) -> np.ndarray:
    """3x3 homogeneous matrix: scale, then rotate (radians), then translate.

    Equivalent to ``translate @ rotate @ scale``.
    """
    c, s = np.cos(rot), np.sin(rot)
    return np.array(
        [
            [sx * c, -sy * s, tx],
            [sx * s, sy * c, ty],
            [0.0, 0.0, 1.0],
        ]
    )


def _flip_matrix(flip_x: bool, flip_y: bool, src_w: int, src_h: int) -> np.ndarray:
    """Matrix that mirrors source pixel coordinates (applied before `rigid`)."""
    return np.array(
        [
            [-1.0 if flip_x else 1.0, 0.0, (src_w - 1) if flip_x else 0.0],
            [0.0, -1.0 if flip_y else 1.0, (src_h - 1) if flip_y else 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# --------------------------------------------------------------------------- #
# TIFF access
# --------------------------------------------------------------------------- #
def _open_tiff(path: Path) -> tuple[np.ndarray, str]:
    """Return (array, axes) for the first series, memory-mapped when possible."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        try:
            data = tifffile.memmap(path, series=0, mode="r")
        except ValueError:
            # Compressed / non-contiguous files can't be memory-mapped.
            logger.warning("%s is not memory-mappable; loading it into RAM.", path.name)
            data = series.asarray()
    return data, axes


def _to_tczyx(data: np.ndarray, axes: str) -> np.ndarray:
    """Return a 5-D (T, C, Z, Y, X) *view* of `data`, whatever its axis order."""
    if data.ndim != len(axes):
        raise ValueError(f"Axes {axes!r} don't match data shape {data.shape}")
    if not axes.endswith("YX"):
        raise ValueError(f"Expected axes ending in 'YX', got {axes!r}")

    # Drop size-1 axes we don't model (e.g. 'Q', 'S', 'I'); refuse real data on them.
    extra = tuple(i for i, a in enumerate(axes) if a not in CANONICAL_AXES)
    for i in extra:
        if data.shape[i] != 1:
            raise ValueError(
                f"Unsupported axis {axes[i]!r} (size {data.shape[i]}) in {axes!r}"
            )
    data = np.squeeze(data, axis=extra)
    axes = "".join(a for a in axes if a in CANONICAL_AXES)

    # Add missing axes as size-1 dimensions, then reorder.
    for a in CANONICAL_AXES:
        if a not in axes:
            data = data[np.newaxis]
            axes = a + axes
    return np.transpose(data, [axes.index(a) for a in CANONICAL_AXES])


# --------------------------------------------------------------------------- #
# Channel loader
# --------------------------------------------------------------------------- #
def load_mif_channel(
    height: int,
    width: int,
    frames: int,
    slices: int,
    path: os.PathLike,
    flipX: bool = False,
    flipY: bool = False,
    scaleX: float = 1.0,
    scaleY: float = 1.0,
    rotXY: float = 0.0,
    transX: float = 0.0,
    transY: float = 0.0,
    useC: int = 1,
    padZ: int = 0,
    batchsize: int = 1,
    interpolation: int = cv2.INTER_LINEAR,
    **kwargs,
) -> Iterator[np.ndarray]:
    """Yield `frames` volumes of shape (slices, height, width) for one channel."""
    path = Path(path)
    if kwargs:
        logger.debug(
            "Ignoring unused channel keys for %s: %s", path.name, sorted(kwargs)
        )

    raw, axes = _open_tiff(path)
    logger.info(
        "src:%s shape:%s dtype:%s axes:%s", path.name, raw.shape, raw.dtype, axes
    )
    data = _to_tczyx(raw, axes)
    n_t, n_c, n_z, src_h, src_w = data.shape

    # ---- validation ------------------------------------------------------- #
    if not 1 <= useC <= n_c:
        raise IndexError(f"{path.name}: channel {useC} requested, file has {n_c}.")
    if padZ < 0 or padZ + slices > n_z:
        raise ValueError(
            f"{path.name}: need padZ ({padZ}) + slices ({slices}) <= Z ({n_z})."
        )
    if n_t != 1 and n_t < frames:
        raise ValueError(f"{path.name}: {frames} frames requested, file has {n_t}.")

    dtype = data.dtype.newbyteorder("=")  # native byte order for OpenCV
    if dtype not in _CV2_DTYPES:
        raise TypeError(
            f"{path.name}: dtype {dtype} is not supported by OpenCV warping."
        )

    # Lazy view: selected channel and Z range only.
    data = data[:, useC - 1, padZ : padZ + slices]

    # Fold flips + rigid transform into one affine matrix (src -> dst).
    M = rigid(scaleX, scaleY, rotXY, transX, transY) @ _flip_matrix(
        flipX, flipY, src_w, src_h
    )
    affine = M[:2]

    def warp(vol: np.ndarray) -> np.ndarray:
        out = np.empty((slices, height, width), dtype=dtype)
        for z, plane in enumerate(vol):
            out[z] = cv2.warpAffine(
                np.ascontiguousarray(plane, dtype=dtype),
                affine,
                (width, height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return out

    # ---- iteration -------------------------------------------------------- #
    if n_t == 1 and frames > 1:
        # Static image: warp once and repeat it for every frame.
        static = warp(np.asarray(data[0]))
        static.flags.writeable = False
        for _ in range(frames):
            yield static
        return

    step = max(1, int(batchsize))
    for t0 in range(0, frames, step):
        batch = np.asarray(data[t0 : min(t0 + step, frames)])  # one read per batch
        for vol in batch:
            yield warp(vol)


# --------------------------------------------------------------------------- #
# MIF parsing
# --------------------------------------------------------------------------- #
def parse_raw_str(x: str):
    """Convert a MIF value to int / float / bool, falling back to str."""
    try:
        ret = float(x)
    except ValueError:
        pass
    else:
        return int(ret) if ret.is_integer() else ret

    lowered = x.lower()
    if lowered in ("true", "t", "y", "yes"):
        return True
    if lowered in ("false", "f", "n", "no"):
        return False
    return x


class MIFLoader:
    """Read a .mif description and stream (C, Z, Y, X) volumes frame by frame."""

    axes = CANONICAL_AXES

    def __init__(self, mif_path: os.PathLike):
        mif_path = Path(mif_path)
        metadata: dict = {}
        channel_props: dict[int, dict] = defaultdict(dict)

        for line in mif_path.read_text().splitlines():
            m = _LINE_RE.match(line)
            if m is None:
                continue
            key, group, raw = m.groups()
            val = parse_raw_str(raw)
            # Keys ending in a number belong to that channel.
            if group is None:
                metadata[key] = val
            else:
                channel_props[int(group)][key] = val

        missing = [k for k in ("width", "height") if k not in metadata]
        if missing:
            raise ValueError(f"{mif_path.name}: missing required keys {missing}")
        if not channel_props:
            raise ValueError(f"{mif_path.name}: no channels defined")

        for ch, props in channel_props.items():
            if "path" not in props:
                raise ValueError(f"{mif_path.name}: channel {ch} has no 'path'")
            p = Path(str(props["path"]))
            props["path"] = p if p.is_absolute() else mif_path.parent / p
            props.setdefault("useC", 1)

        width = int(metadata["width"])
        height = int(metadata["height"])
        slices = int(metadata.get("slices", 1))
        frames = int(metadata.get("frames", 1))

        self.metadata = metadata
        self.channel_props = dict(sorted(channel_props.items()))
        # T, C, Z, Y, X
        self.shape = (frames, len(self.channel_props), slices, height, width)

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter()

    def iter(self, order: Sequence[int] | None = None) -> Iterator[np.ndarray]:
        """Yield one (C, Z, Y, X) stack per frame."""
        order = list(self.channel_props) if order is None else list(order)
        unknown = [ch for ch in order if ch not in self.channel_props]
        if unknown:
            raise KeyError(
                f"Unknown channel(s) {unknown}; available: {list(self.channel_props)}"
            )

        T, _, Z, Y, X = self.shape
        loaders = [
            load_mif_channel(
                width=X, height=Y, frames=T, slices=Z, **self.channel_props[ch]
            )
            for ch in order
        ]
        for vols in zip(*loaders):
            yield np.stack(vols, axis=0)
