import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import tifffile

# Load the TIFF image stack and save as a chunked Zarr array to disk
def tiff_to_zarr(tiff_file:str="/n/groups/htem/users/yazatian/xray/cutouts/monkeyv1axonseg002/img/vol.tiff",
                out_file:str="../../xray-challenge-entry/data/monkey_xnh.zarr",
                out_ds:str="volumes/training_raw_003") -> None:
    tiff_stack: np.ndarray = tifffile.imread(tiff_file)
    tiff_stack = np.transpose(tiff_stack, (2, 1, 0))
    print(type(tiff_stack), tiff_stack.shape)

    roi: Roi = Roi(offset=(0, 0, 0), shape=Coordinate(100000,100000,100000))
    print("Roi: ", roi)
    voxel_size: Coordinate = Coordinate(100, 100, 100)

    ds = prepare_ds(
        filename=out_file,
        ds_name=out_ds,
        total_roi=roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        delete=True,
    )

    ds[roi] = tiff_stack

    print("Image stack saved as Zarr dataset.")

if __name__=="__main__":
    tiff_to_zarr(tiff_file="/n/groups/htem/users/yazatian/xray/cutouts/monkeyv1axonseg002/img/vol.tiff",
                 out_ds="volumes/training_raw_002")