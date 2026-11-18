def select_compute_device(cuda_available: bool) -> int | str:
    return 0 if cuda_available else "cpu"
