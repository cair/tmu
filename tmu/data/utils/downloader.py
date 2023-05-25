from pathlib import Path
import hashlib
import logging

_LOGGER = logging.getLogger(__name__)


try:
    from tqdm import tqdm
    import requests
except ModuleNotFoundError as e:
    _LOGGER.exception("Missing packages tqdm and requests. Install with 'pip install requests tqdm'")
    raise e



def get_file(path, origin, file_hash):
    # determine the cache directory
    cache_dir = Path.home() / '.cache' / 'datasets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / path

    download = False
    if fpath.exists():
        # File already exists, check hash
        file_tmp = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if file_hash != file_tmp:
            _LOGGER.info('Hash mismatch, redownloading file')
            download = True
    else:
        download = True

    if download:
        _LOGGER.info('Downloading data from', origin)
        r = requests.get(origin, stream=True)
        total_size = int(r.headers.get('content-length', 0))

        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(fpath, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            _LOGGER.info("ERROR, something went wrong")

    return str(fpath)


if __name__ == "__main__":
    import numpy as np
    origin_folder = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    )
    path = get_file(
        "mnist.npz",
        origin=origin_folder + "mnist.npz",
        file_hash=(  # noqa: E501
            "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
        ),
    )
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
        print((x_train, y_train), (x_test, y_test))
        print(path)
