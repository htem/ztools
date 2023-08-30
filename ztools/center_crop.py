from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate


def make_cutout(
    out_file: str,
    out_dataset: str,
    src_file: str,
    src_dataset: str,
) -> bool:
    src = open_ds(src_file, src_dataset)
    print("Roi: ", src.roi)
    print("Shape: ", src.shape)

    roi = get_centered_roi(src.data_roi, Coordinate(1000, 1000, 1000), src.voxel_size)
    print("Centered Roi: ", roi)
    roi = roi.snap_to_grid(src.voxel_size, mode="closest")
    print("Snapped Roi:", roi)
    out = prepare_ds(out_file, out_dataset, roi, src.voxel_size, src.dtype, delete=True)
    # src.roi.center
    out[roi] = src[roi].to_ndarray()
    return True


def get_centered_roi(data_roi: Roi, size: Coordinate, voxel_size: Coordinate) -> Roi:
    data_center: Coordinate = data_roi.get_center()
    print("Data center: ", data_center)
    centered_roi: Roi = (
        Roi(data_center, (0, 0, 0))
        .grow(size / 2, size / 2)
        .snap_to_grid(voxel_size, "grow")
    )
    return centered_roi


if __name__ == "__main__":
    make_cutout(
        out_file="../tomo_vol.zarr",
        out_dataset="mouse_cb_full_inverted_crop",
        src_file="../tomo_vol.zarr",
        src_dataset="mouse_cb_full_inverted",
    )
