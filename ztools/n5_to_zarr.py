import numpy as np
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

# Load the TIFF image stack and save as a chunked Zarr array to disk
def n5_to_zarr(n5_file:str,
                 n5_ds:str,
                out_file:str="../../xray-challenge-entry/data/monkey_xnh.zarr",
                out_ds:str="volumes/training_raw_003") -> None:
    store: np.ndarray = open_ds(n5_file, n5_ds).data
    store = np.transpose(store, (2, 1, 0))
    print(type(store), store.shape)

    roi: Roi = Roi(offset=(0, 0, 0), shape=Coordinate(1024*33, 1024*33, 1024*33)) # desired shape * voxel size
    print("Roi: ", roi)
    voxel_size: Coordinate = Coordinate(33, 33, 33)

    ds = prepare_ds(
        filename=out_file,
        ds_name=out_ds,
        total_roi=roi,
        voxel_size=voxel_size,
        dtype=np.uint8,
        delete=True,
    )

    ds[roi] = store

    print("Image stack saved as Zarr dataset.")

if __name__=="__main__":
    n5_to_zarr(n5_file="/n/groups/htem/users/br128/xray-challenge-entry/setups/enhancement/gt_enhanced.n5",
               n5_ds="volumes/raw_30nm",
               out_file="/n/groups/htem/users/br128/xray-challenge-entry/setups/enhancement/enhanced_gt.zarr",
               out_ds="volumes/raw_30nm")