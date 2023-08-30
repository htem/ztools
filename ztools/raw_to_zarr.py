import numpy as np
from funlib.geometry import Roi
from funlib.persistence import prepare_ds, Array
import daisy


def raw_to_zarr(
    in_file_path: str = "",
    out_file_path: str = "",
    ds_name: str = "",
    volume_size: list[int, int, int] = [1000]*3,
    voxel_size: list[int, int, int] = [10]*3,
    dtype_in=np.uint8,
    dtype_out=np.uint8,
) -> None:
    
    # set Zarr total ROI, along with Daisy read/write ROIs
    total_roi: Roi= Roi(offset=(0, 0, 0), shape=np.array(volume_size) * np.array(voxel_size))

    print(f"Opening a zarr volume")
    vol: Array = prepare_ds(
        filename=out_file_path,
        ds_name=ds_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=dtype_out,
        delete=True,
    )

    # load raw file into memory and reshape according to the desired Zarr volume size
    print("Loading raw volume into memory...")
    im: np.ndarray = np.fromfile(file=in_file_path, dtype=dtype_in)

    print("Reshaping...")
    im: np.ndarray = im.reshape(*volume_size, order="F").astype(dtype=dtype_out)

    # check compatible shapes
    assert vol.shape == im.shape

    # write and save the array to the Zarr
    print("Saving to Zarr...")
    vol[total_roi] = im
    print("Finished saving to Zarr.")


def raw_to_zarr_blockwise(
    in_file_path: str = "",
    out_file_path: str = "",
    ds_name: str = "",
    volume_size: list[int, int, int] = [1000]*3,
    voxel_size: list[int, int, int] = [10]*3,
    dtype_in=np.uint8,
    dtype_out=np.uint8,
) -> bool:
    # set Zarr total ROI, along with Daisy read/write ROIs
    total_roi = read_roi = write_roi = Roi((0, 0, 0), np.array(volume_size) * np.array(voxel_size))

    print(f"Opening a zarr volume")
    vol: Array = prepare_ds(
        filename=out_file_path,
        ds_name=ds_name,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=dtype_out,
        delete=True,
    )

    # load raw file and reshape according to the desired Zarr volume size
    print("Loading raw volume into memory...")
    im: np.ndarray = np.fromfile(file=in_file_path, dtype=dtype_in)

    print("Reshaping...")
    im: np.ndarray = im.reshape(*volume_size, order="F").astype(dtype=dtype_out)

    # daisy worker to write chunks of the in-memory array to the Zarr
    def write_worker(
        block: daisy.Block, im: np.ndarray = im, vol: Array = vol
    ) -> bool:
        im: np.ndarray = im[block.read_roi]
        vol[block.write_roi] = im
        return True
    
    # check compatible shapes
    assert vol.shape == im.shape

    # create a Daisy distributed task
    task: daisy.Task = daisy.Task(
        task_id="ConvertZarrTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=write_worker,
        num_workers=25,
        read_write_conflict=False,
        fit="valid",
    )

    # write and save the array to the Zarr, blockwise
    print("Saving to Zarr...")
    done: bool = daisy.run_blockwise(tasks=[task])
    print("Finished saving Zaarr.")
    return done


def get_raw_size() -> tuple:
    pass

def get_valid_voxel_size() -> tuple:
    pass