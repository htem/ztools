import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import mrcfile

# Load the TIFF image stack
arr = None
with mrcfile.open("../mouse_cerebellum_tem_tomo.mrc") as mrc:
    arr = mrc.data
print(arr.shape)

arr = np.invert(arr)

roi: Roi = Roi(offset=(0,0,0), shape=Coordinate(arr.shape))
print("Roi: ", roi)
voxel_size: Coordinate = Coordinate(1, 1, 1)

ds = prepare_ds(filename="../tomo_vol.zarr",
                    ds_name="mouse_cb_full_inverted",
                    total_roi=roi,
                    voxel_size=voxel_size,
                    dtype=np.uint8,
                    delete=True)

ds[roi] = arr

print("Image stack saved as Zarr dataset.")
