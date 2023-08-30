import numpy as np
from funlib.persistence import prepare_ds, Array
from funlib.geometry import Coordinate, Roi
import mrcfile

# Load the TIFF image stack
def mrc_to_zarr(in_file_path:str=None,
                out_file_path:str=None,
                out_dataset:str=None,
                voxel_size:tuple=(1,1,1),
                invert:bool=True) -> None:
    # set space and load MRC data into a numpy array
    arr: np.ndarray = None
    with mrcfile.open(name=in_file_path) as mrc:
        arr: np.ndarray = mrc.data

    print("MRC Data Shape: ",arr.shape)

    if invert:
        arr: np.ndarray = np.invert(arr)

    # set Zarr total ROI given the in-memory data shape
    roi: Roi = Roi(offset=(0, 0, 0), shape=Coordinate(arr.shape))
    print("Roi: ", roi)

    # cast the input voxel size as a Coordinate
    voxel_size: Coordinate = Coordinate(voxel_size)

    # create/open on-disk Zarr to write to
    ds: Array = prepare_ds(
        filename=out_file_path,
        ds_name=out_dataset,
        total_roi=roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        delete=True,
    )

    # write in-memory array to the established Zarr
    ds[roi] = arr

    print("Image stack saved as Zarr dataset.")

def get_mrc_size() -> tuple:
    pass

def get_valid_voxel_size() -> tuple:
    pass
