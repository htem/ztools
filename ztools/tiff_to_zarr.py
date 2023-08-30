import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import tifffile

# Load the TIFF image stack
tiff_stack = tifffile.imread('../tomo_vol.tif')

print(type(tiff_stack))
cropped_array = tiff_stack[0:2000, 0:200, 0:2000]

print(cropped_array.shape)

roi: Roi = Roi(offset=(0,0,0), shape=Coordinate(cropped_array.shape))
print("Roi: ", roi)
voxel_size: Coordinate = Coordinate(1, 1, 1)

ds = prepare_ds(filename="../tomo_vol.zarr",
                    ds_name="reslice_8bit_tomo1_vol_alignedv16_imaps",
                    total_roi=roi,
                    voxel_size=voxel_size,
                    dtype=np.uint8,
                    delete=True)

ds[roi] = cropped_array

print("Image stack saved as Zarr dataset.")
