import logging
import hashlib
from datetime import datetime
from pathlib import Path
import tarfile
import zipfile

_LOGGER = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    import requests
except ModuleNotFoundError as e:
    _LOGGER.exception("Missing packages tqdm and requests. Install with 'pip install requests tqdm'")
    raise e


def get_file(path, origin, file_hash, extract=False, extract_archive_format='auto'):
    # determine the cache directory
    cache_dir = Path.home() / '.cache' / 'tmu' / 'datasets'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / path

    extraction_indicator = fpath.with_suffix(fpath.suffix + '.extracted')  # indicator file

    download = False
    if fpath.exists():
        # File already exists, check hash
        file_tmp = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if file_hash != file_tmp:
            _LOGGER.error('Hash mismatch, redownloading file')
            download = True
    else:
        download = True

    if download:
        _LOGGER.info(f'Downloading data from {origin}')
        if "dropbox" in origin:
            headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        else:
            headers = {}
        r = requests.get(origin, stream=True, allow_redirects=True, headers=headers)
        total_size = int(r.headers.get('content-length', 0))

        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(fpath, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            _LOGGER.error("ERROR, something went wrong")

    if fpath.suffix == ".npz":
        return str(fpath)

    # If it's a .tar.gz file, we want the name without both extensions.
    if fpath.suffix == '.gz' and fpath.stem.endswith('.tar'):
        extracted_dir = fpath.parent / fpath.stem[:-4]  # Removing .tar from the stem
    else:
        extracted_dir = fpath.parent / fpath.stem  # Path to extracted directory

    if extract and not extraction_indicator.exists():

        if extract_archive_format == 'auto':
            if fpath.suffix == '.zip':
                extract_archive_format = 'zip'
            elif fpath.suffix in ['.tar', '.gz']:
                extract_archive_format = 'tar'

        if extract_archive_format == 'zip':
            with zipfile.ZipFile(fpath, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
        elif extract_archive_format == 'tar':
            with tarfile.open(fpath, 'r:gz') as tar_ref:
                tar_ref.extractall(extracted_dir)
        else:
            raise ValueError(f'Unknown archive format {extract_archive_format}')

        with open(extraction_indicator, 'w') as f:
            f.write('Extraction completed on ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        return str(extracted_dir)  # Return path to extracted directory
    else:
        return str(extracted_dir)


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
