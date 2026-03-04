from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Generator, Optional, Sequence

import cv2
import numpy as np
import tifffile


def rigid(sx: float, sy: float, rot: float, tx: float, ty: float) -> np.ndarray:
    scale = np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    cos, sin = np.cos(rot), np.sin(rot)
    rotate = np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translate = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ]
    )

    return translate @ rotate @ scale


def load_mif_channel(
    height: int,
    width: int,
    frames: int,
    slices: int,
    path: os.PathLike,
    flipX: bool,
    flipY: bool,
    scaleX: float,
    scaleY: float,
    rotXY: float,
    transX: float,
    transY: float,
    numC: int,
    useC: int,
    padZ: int = 0,
    batchsize: int = 1,
    **kwargs,
):
    with tifffile.TiffFile(path) as tif:
        axes = tif.series[0].axes.upper()

    assert axes.endswith("YX"), axes

    data = tifffile.memmap(path, mode="r")
    shape = data.shape

    # Instead of: assert len(axes) == len(shape)
    # Try forcing them to match:
    if len(axes) != len(shape):
        # Common issue: tifffile adds axes for dimensions of size 1
        # that memmap might have stripped.
        raise ValueError(
            f"Metadata axes {axes} ({len(axes)}) don't match data shape {shape} ({len(shape)})"
        )

    axes_to_idx = {c: i for i, c in enumerate(axes)}
    dims = {c: shape[i] for i, c in enumerate(axes)}

    target_c_idx = useC - 1
    if 0 > target_c_idx or target_c_idx > dims.get("C", 1):
        raise IndexError(
            f"Channel {useC} requested, but only {dims.get('C')} channels exist."
        )

    actual_z = dims.get("Z", 1)
    expected_z = padZ + actual_z
    if slices != expected_z:
        raise ValueError(
            f"Z-Slice Mismatch: MIF expects {slices}, calculation gives {expected_z}"
        )
    M = rigid(scaleX, scaleY, rotXY, transX, transY)
    M_inv = np.linalg.inv(M)

    # Create a grid of (x, y) coordinates
    grid = np.indices((height, width), dtype=np.float32)

    # To get your (3, N) matrix for the affine transformation:
    coords = np.vstack(
        [
            grid[1].reshape(1, -1),  # X
            grid[0].reshape(1, -1),  # Y
            np.ones((1, height * width), dtype=np.float32),
        ]
    )
    # 2. Apply your affine/perspective matrix M to the grid
    # M is your 3x3 matrix
    transformed_coords = M_inv @ coords
    # Perspective division
    map_x = (
        (transformed_coords[0] / transformed_coords[2])
        .reshape(height, width)
        .astype(np.float32)
    )
    map_y = (
        (transformed_coords[1] / transformed_coords[2])
        .reshape(height, width)
        .astype(np.float32)
    )

    idx = np.arange(frames, dtype=int)
    batched_idx = np.array_split(idx, max(1, (frames // batchsize)))

    for batch in batched_idx:
        # we read a batch of data to reduce the io access
        if "T" in axes_to_idx:
            batch_data = np.take(data, axis=axes_to_idx["T"], indices=batch)
        else:
            batch_data = data[None, ...]

        if "C" in dims:
            # extract C from C
            assert dims["C"] >= useC, f"{useC} is invalid"
            batch_data = np.take(batch_data, axis=axes_to_idx["C"], indices=useC - 1)

        if flipX:
            batch_data = np.flip(batch_data, -1)
        if flipY:
            batch_data = np.flip(batch_data, -2)

        for src in batch_data:

            # wrapping single channel in list to make it iterable.
            if "Z" not in dims:
                src = [src]

            # 3. Apply the map to all slices in the Z-stack
            # This moves the loop into C++ internal logic
            final_z_stack = np.zeros((expected_z, height, width), dtype=src.dtype)
            for i, slice_z in enumerate(src):
                cv2.remap(
                    src=slice_z,
                    map1=map_x,
                    map2=map_y,
                    interpolation=cv2.INTER_CUBIC,
                    dst=final_z_stack[i + padZ],
                    borderMode=cv2.BORDER_REPLICATE,
                )

            yield final_z_stack


def parse_raw_str(x: str):
    try:
        ret = float(x)
        return int(ret) if ret.is_integer() else ret
    except ValueError:
        pass

    true_flags = ("true", "t", "1", "y", "yes")
    false_flags = ("false", "f", "0", "n", "no")
    if x.lower() in true_flags or x.lower() in false_flags:
        return x.lower() in true_flags

    return x


class MIFLoader:
    def __init__(self, mif_path: os.PathLike):
        mif_path = Path(mif_path)
        mif_obj = mif_path.open("r")
        metadata = {}
        channel_props = defaultdict(dict)
        regex = re.compile(r"^([a-zA-Z]+)(\d+)?\s*=\s*(\S+);$")
        for line in mif_obj.readlines():
            res = regex.match(line)
            if res is None:
                continue
            key, gp, val = res.group(1), res.group(2), res.group(3)

            val = parse_raw_str(val)
            # Only the attributes end with number will be assigned to channels group.
            if gp is None:
                metadata[key] = val
            else:
                channel_props[gp][key] = val

        for k in channel_props:
            p = Path(channel_props[k]["path"])

            if not p.is_absolute():
                # If the file is not point to the correct
                p = mif_path.parent.joinpath(p)
            channel_props[k]["path"] = p
            channel_props[k]["useC"] = channel_props[k].get("useC", 1)

        width = metadata["width"]
        height = metadata["height"]
        slices = metadata.get("slices", 1)
        frames = metadata.get("frames", 1)

        # T, C, Z, Y, X
        self.shape = (frames, len(channel_props), slices, height, width)
        self.axes = "TCZYX"
        self.channel_props = {int(k): v for k, v in channel_props.items()}

    def iter(self, order: Optional[Sequence[int]] = None) -> Generator[np.ndarray]:
        """Yield the image stack with axes (C, Z, Y, X)"""
        if order is None:
            order = sorted(self.channel_props.keys())
        T, C, Z, Y, X = self.shape
        loaders = [
            load_mif_channel(
                width=X, height=Y, frames=T, slices=Z, **self.channel_props[ch]
            )
            for ch in order
        ]
        for vol in zip(*loaders):
            vol = np.stack(vol, axis=0)
            yield vol
