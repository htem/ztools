from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate, Array


def center_crop(
    out_file: str,
    out_dataset: str,
    src_file: str,
    src_dataset: str,
) -> bool:
    src: Array = open_ds(filename=src_file, ds_name=src_dataset)
    print("Roi: ", src.roi)
    print("Shape: ", src.shape)

    def get_centered_roi(data_roi: Roi, size: Coordinate, voxel_size: Coordinate) -> Roi:
        data_center: Coordinate = data_roi.get_center()
        print("Data center: ", data_center)
        centered_roi: Roi = (
            Roi(offset=data_center, shape=(0, 0, 0))
            .grow(amount_neg=size / 2, amount_pos=size / 2)
            .snap_to_grid(voxel_size=voxel_size, mode="grow")
        )
        return centered_roi

    roi: Roi = get_centered_roi(data_roi=src.data_roi, size=Coordinate(1000, 1000, 1000), voxel_size=src.voxel_size)
    print("Centered Roi: ", roi)

    roi = roi.snap_to_grid(voxel_size=src.voxel_size, mode="closest")
    print("Snapped Roi:", roi)
    out: Array = prepare_ds(filename=out_file, 
                            ds_name=out_dataset, 
                            total_roi=roi, 
                            voxel_size=src.voxel_size, 
                            dtype=src.dtype,
                            delete=True)
    out[roi] = src[roi].to_ndarray()
    return True
