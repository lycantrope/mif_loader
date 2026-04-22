import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin

from mif_loader import MIFLoader

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


GRID_SPACING_ZYX = (1.0, 0.4, 0.4)  # unit: micrometer per pixel


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", help="MIF file path (*.mif)", required=True)
    parser.add_argument(
        "--skip_z", default=0, type=int, help="Number of z frame to skip."
    )

    args = parser.parse_args()

    mif_path = Path(args.input)

    if not mif_path.is_file() or not mif_path.suffix == ".mif":
        parser.error(f"Input file is is not a valid mif file. {args.input}")

    skip_z = args.skip_z
    # Create lazy loader to parse data from MIF-Tiff.
    mif_loader = MIFLoader(mif_path)
    T, C, Z, Y, X = mif_loader.shape
    assert skip_z < Z, f"Number of z-skip cannot be greater than z size: {skip_z:d}"
    print(f"Converting {mif_path.name} into HDF file")
    with h5py.File(mif_path.with_suffix(".h5"), mode="w") as handler:
        handler.attrs["frames"] = T
        handler.attrs["channels"] = C
        handler.attrs["slices"] = Z - skip_z
        handler.attrs["width"] = X
        handler.attrs["height"] = Y
        for t, im_stack in enumerate(mif_loader.iter()):
            # CZYX, we skip the first two frames.
            print(im_stack.shape)
            im_stack = im_stack[:, skip_z:, :, :]
            group = handler.create_group(
                f"t{t:d}",  # t0
                track_order=True,
                track_times=True,
            )
            group.attrs["axis_order"] = "ZYX"
            for c, im in enumerate(im_stack):
                # Writing data into disk
                ds = group.create_dataset(
                    f"c{c:d}",  # c0
                    data=im,
                    # **hdf5plugin.Zstd(),
                )
                ds.attrs["element_size_um"] = GRID_SPACING_ZYX

    print(f"Successfully conversion of mif to hdf file at: {str(mif_path.parent)}")


if __name__ == "__main__":
    main()
