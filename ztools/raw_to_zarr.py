import numpy as np
from funlib.geometry import Roi
from funlib.persistence import prepare_ds
from funlib.geometry import Roi, Coordinate
from funlib.persistence import prepare_ds, Array
import daisy


def save_volume_to_zarr(in_file_path:str="/n/data3/hms/neurobio/htem/xnh/volraw/s46_V1_100nm_7_q3_rec_.raw", 
                        out_file_path:str="../../data/monkey_xnh.zarr", 
                        ds_name:str="s46_V1_100nm_7_q3_rec",
                        volume_size: list=[3216, 3216, 3216], 
                        voxel_size:list=[10, 10, 10], 
                        dtype_in=np.uint8, 
                        dtype_out=np.uint8) -> None:
    total_roi = Roi((0, 0, 0), np.array(volume_size) * np.array(voxel_size))


    read_roi =write_roi = Roi(offset=(0,) * 3, shape=Coordinate((10,10,10)))*Coordinate(voxel_size)

    print(f'Opening a zarr volume')
    vol: Array = prepare_ds(filename=out_file_path,
                           ds_name=ds_name,
                           total_roi=total_roi,
                           voxel_size=voxel_size,
                           dtype=dtype_out,
                           delete=True)

    # --- Load image volume and upload --- #
    print('Loading raw volume into memory...')

    im: np.ndarray = np.fromfile(file=in_file_path, dtype=dtype_in)
    print('Reshaping...')
    im: np.ndarray = im.reshape(*volume_size, order='F').astype(dtype=dtype_out)

    def zarr_write_worker(block:daisy.Block, im:np.ndarray=im, vol:np.ndarray=vol) -> bool:
        im: np.ndarray = im[block.read_roi]
        vol[block.write_roi] = im
        return True

    assert vol.shape == im.shape
    task: daisy.Task = daisy.Task(task_id="ConvertZarrTask",
                                total_roi=total_roi,
                                read_roi=read_roi,
                                write_roi=write_roi,
                                process_function=zarr_write_worker,
                                num_workers=25,
                                read_write_conflict=False,
                                fit="valid",)


    print('Saving to zarr...')
    # done: bool = daisy.run_blockwise(tasks=[task])
    vol[total_roi] = im
    print('Done.')