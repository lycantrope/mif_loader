from __future__ import annotations

import argparse
import functools
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import roifile
from scipy import optimize



N_CPU = os.cpu_count() or 1


def rigid(x: np.ndarray) -> np.ndarray:
    sx, sy, rot, tx, ty = x
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


def init_params() -> np.ndarray:
    init = np.random.rand(5)
    # The scale_x and scale_y were initiated from 1.0
    init[:2] = 1.0
    return init


def evaluate(param, _from, _to):
    return np.sum((_to - rigid(param) @ _from) ** 2)


def run_trial(trial_id, pos2h, pos1h):
    np.random.seed(trial_id + 42)
    res = optimize.fmin(
        evaluate,
        init_params(),
        args=(pos2h, pos1h),
        disp=False,
        full_output=True,
        ftol=1e-6,
    )
    return res[:2]


def timer(func):
    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        tic = time.perf_counter()
        ret = func(*arg, **kwargs)
        toc = time.perf_counter()
        print(f"Elapsed: {toc-tic:.2f} (sec)")
        return ret

    return wrapper


@timer
def find_2d_affine_parameters(file: pathlib.Path, n_trial:int=100):
    roilist = roifile.roiread(file)

    if not isinstance(roilist, list) or len(roilist) % 2 != 0:
        raise ValueError(
            "The Roi file only contains odd number of Roi: {}".format(file)
        )

    if any((roi.roitype != roifile.ROI_TYPE.POINT for roi in roilist)):
        raise TypeError("Please use point tools to select roi for alignment")

    rois = np.vstack([(roi.coordinates()) for roi in roilist])
    ones = np.ones(len(roilist))[:, None]
    rois = np.hstack((rois, ones))
    # The ROIs from the two images must be interleaved
    # [pos11, pos21, pos12, pos22, pos13, pos23, ..., pos1n, pos2n]
    pos1h = rois[::2].T
    pos2h = rois[1::2].T

    with ProcessPoolExecutor(min(N_CPU, n_trial)) as pool:
        # We 'zip' the changing trial index with the constant arrays

        res1 = [pool.submit(run_trial, i, pos2h, pos1h) for i in range(n_trial)]

        # Find affine parameters from pos1 to pos2
        params1_best, fval1_min = min(
            (res.result() for res in as_completed(res1)),
            key=lambda x: x[1],
        )
        # Find affine parameters from pos2 to pos1
        res2 = [pool.submit(run_trial, i, pos1h, pos2h) for i in range(n_trial)]

        # Find affine parameters from pos1 to pos2
        params2_best, fval2_min = min(
            (res.result() for res in as_completed(res2)),
            key=lambda x: x[1],
        )

    fmt = """Ch{_from:d} -> Ch{to:d}
scaleX{_from:d} = {:f};
scaleY{_from:d} = {:f};
rotXY{_from:d} = {:f};
transX{_from:d} = {:f};
transY{_from:d} = {:f};"""
    print(fmt.format(*params2_best, _from=1, to=2))
    print(fmt.format(*params1_best, _from=2, to=1))


def main():
    parser = argparse.ArgumentParser("mif_registor")
    parser.add_argument(
        "filename",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-n_trial",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    try:
        find_2d_affine_parameters(args.filename, args.n_trial)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
