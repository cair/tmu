import abc
import pathlib
import shutil
import sys
import tempfile
import typing
from typing import Dict
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn
from urllib.request import urlopen
import json
import platform
from pathlib import Path
import os
import time
import logging
import io
import zipfile
import tarfile
import lzma
import pandas as pd
import importlib
import importlib.util

_LOGGER = logging.getLogger(__name__)


class TMUDatasetSource:
    _metadata_url: str = "https://api.github.com/repos/cair/tmu-datasets"
    _tmu_dataset_name: str = "tmu-datasets"

    def __init__(self):
        pass

    def _get_config_dir(self, cache_dir):
        if cache_dir is not None:
            cdir = pathlib.Path(cache_dir)
            cdir.mkdir(exist_ok=True)
            return cdir

        system = platform.system()
        if system == 'Windows':
            conf_dir = Path(os.environ['APPDATA']) / self._tmu_dataset_name
        elif system == 'Darwin':
            conf_dir = Path.home() / 'Library' / 'Preferences' / self._tmu_dataset_name
        else:  # Linux, etc.
            conf_dir = Path.home() / '.config' / self._tmu_dataset_name
        conf_dir.mkdir(exist_ok=True)
        return conf_dir

    def _get_releases(self, cache, cache_max_age, cache_dir) -> list:
        config_dir = self._get_config_dir(cache_dir=cache_dir)
        release_file = config_dir / "releases.json"
        _LOGGER.debug(f"Found config directory at: {config_dir}. Release File: {release_file}")
        _LOGGER.debug(f"Release file exists={release_file.exists()}")

        # Reading from cache
        if cache and release_file.exists():

            cache_age = (time.time() - release_file.stat().st_mtime) / 60
            _LOGGER.debug(f"Release file age: {cache_age} minutes")

            if cache_age <= cache_max_age:
                with release_file.open("rb") as f:
                    releases = json.load(f)
                    _LOGGER.debug(f"Loading {release_file} from file is successful")
                    return releases

            _LOGGER.debug(f"Cache is out of date..")

        # Reading from GitHub
        with urlopen(f"{self._metadata_url}/releases") as response:

            if response.code != 200:
                raise RuntimeError(f"Could not connect github at {self._metadata_url}")

            data = response.read()
            releases = json.loads(data)
            _LOGGER.debug(f"Writing release data to {release_file}")
            with release_file.open("w+") as f:
                json.dump(releases, f)

            assert len(releases) > 0, "There must be at least 1 release in tmu-datasets"
        return releases

    def _get_latest_release(self, releases: list):
        releases_cpy = releases.copy()
        sorted(releases_cpy, key=lambda item: item["id"])
        return releases_cpy[0]

    def _download_release_archive(self, archive_url, target_dir) -> pathlib.Path:
        response = urlopen(archive_url)

        total_size = int(response.info().get("Content-Length", 0))
        block_size = 1024

        f = io.BytesIO()

        try:
            import tqdm
            status = tqdm.tqdm(total=total_size // block_size, unit="KB", desc="tmu-datasets")
        except ImportError:
            status = None

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            n_read = len(buffer)

            if status is not None:
                status.update(n_read // block_size)

        with tempfile.TemporaryDirectory(prefix="tmu-datasets-") as td:
            try:
                f.seek(0)
                archive_zip = zipfile.ZipFile(f)
                archive_zip.extractall(str(td))
            except zipfile.BadZipfile:
                f.seek(0)
                with tarfile.open(fileobj=f, mode="r:gz") as tar:
                    tar.extractall(str(td))

            # Move extracted files into root of temp directory
            temp_dir_pathlib = Path(str(td))

            root_dir = next(temp_dir_pathlib.glob("cair-tmu-datasets*"))
            for item in root_dir.glob("*"):
                shutil.move(item, str(td))
            root_dir.rmdir()

            # Finally move all dataset items to target dir
            target_dir.mkdir(exist_ok=True)
            _LOGGER.debug(f"Moving {temp_dir_pathlib} to {target_dir}")
            for item in temp_dir_pathlib.glob("*"):
                shutil.move(item, target_dir)

        return target_dir

    def _download_dataset(self, latest_release_metadata, cache_dir):
        system = platform.system()
        if system == "Windows":
            ball_url_key = "zipball_url"
        else:
            ball_url_key = "tarball_url"

        dl_url = latest_release_metadata[ball_url_key]
        dl_name = latest_release_metadata["name"]

        config_dir = self._get_config_dir(cache_dir=cache_dir)
        dataset_dir = config_dir / dl_name

        if dataset_dir.exists():
            _LOGGER.debug(f"Dataset directory {dataset_dir} already exists.")
            return dataset_dir

        _LOGGER.debug(f"Dataset directory {dataset_dir} not found. Downloading release {dl_name}")
        temp_dataset_dir = self._download_release_archive(dl_url, target_dir=dataset_dir)




        return dataset_dir

    def _get_metadata(self, metadata, key, default, extra_info=None):
        if key not in metadata:
            _LOGGER.warning(f"Malformed metadata.json. Key={key}, Extra={extra_info}.")
            return default
        return metadata[key]

    def _parse_dataset_v1(self, metadata: dict, dataset_path: pathlib.Path, cache):
        dataset_file = dataset_path / metadata["file"]
        dataset_name = dataset_file.name

        dataset_type = self._get_metadata(metadata, "type", "table", "'type'")
        opt = self._get_metadata(metadata, "options", {}, "'options'")

        if dataset_type == "table":
            option_separator = self._get_metadata(opt, "separator", ",", "'options.separator'")
            option_header = self._get_metadata(opt, "header", 1, "'options.header'")

            if dataset_name.endswith(".txt.xz"):
                # The dataset location when decompressed
                dataset_decompressed_path = dataset_file.with_suffix("")

                # Source to read dataset from
                read_source = dataset_decompressed_path

                # If decompressed version does not exist and cache is enabled
                if not dataset_decompressed_path.exists() or not cache:
                    # Open compressed version and write to decompressed file
                    read_source = lzma.open(dataset_file)
                    with dataset_decompressed_path.open("wb+") as f:
                        f.write(read_source.read())
                        read_source.seek(0)
                elif cache:
                    read_source = dataset_decompressed_path.open("rb+")

                df = pd.read_csv(
                    read_source,
                    sep=option_separator,
                    header=0 if option_header else None
                )

                if hasattr(read_source, "close"):
                    read_source.close()

                return df
        else:
            error_msg = f"Found dataset type {dataset_type} which is not supported!"
            _LOGGER.error(error_msg)
            raise RuntimeError(error_msg)

    @staticmethod
    def _load_module_from_file(file_path: str, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _preprocess(self, preprocess_metadata, dataset_path, dataset):
        preprocess_enabled = self._get_metadata(preprocess_metadata, "enabled", False, "preprocess.enabled")
        if not preprocess_enabled:
            return dataset

        preprocess_type = self._get_metadata(preprocess_metadata, "type", None, "preprocess.type")
        supported_preprocess_types = ["file"]

        if preprocess_type not in supported_preprocess_types:
            error = f"Preprocess type not supported for {preprocess_metadata}. Supported: {supported_preprocess_types}. " \
                    f"Found: {preprocess_type}"
            _LOGGER.error(error)
            raise RuntimeError(error)

        if preprocess_type == "file":
            preprocess_filename = self._get_metadata(preprocess_metadata, "filename", None, "preprocess.filename")
            if not preprocess_filename:
                error = "Missing preprocess.filename in metadata.json"
                _LOGGER.error(error)
                raise RuntimeError(error)

            preprocess_file = dataset_path / preprocess_filename
            if not preprocess_file.exists():
                error = f"Could not find the file {preprocess_file}. preprocess.filename was not found in {dataset_path}."
                _LOGGER.error(error)
                raise RuntimeError(error)

            pp_module = TMUDatasetSource._load_module_from_file(preprocess_file, "PreprocessModule")
            return pp_module.preprocess(dataset)

    def _preprocess_dataset(self, metadata, dataset_path, dataset):
        opt_preprocess = self._get_metadata(metadata, "preprocess", None, "Could not find 'preprocess' in metadata")
        if opt_preprocess is not None:
            dataset = self._preprocess(opt_preprocess, dataset_path, dataset)

        return dataset

    def _get_metadata_file(self, dataset_path):
        metadata_file = dataset_path / "metadata.json"

        if not metadata_file.exists():
            err = f"Could not find metadata.json for dataset \"{dataset_path.name}\". All datasets must have a " \
                  f"metadata.json file. Please consult " \
                  "documentation!"
            _LOGGER.warning(err)
            raise RuntimeError(err)

        with metadata_file.open("rb+") as mdf:
            metadata = json.load(mdf)

        return metadata

    def _extract_dataset(self, metadata, dataset_path: pathlib.Path, cache):

        supported_versions = [1]  # Supported versions

        if metadata["version"] not in supported_versions:
            err = f"Invalid version: {metadata['version']} for {dataset_path}"
            _LOGGER.error(err)
            raise RuntimeError(err)

        parse_fn = getattr(self, f"_parse_dataset_v{metadata['version']}")

        # Finally, parse dataset
        dataset = parse_fn(metadata, dataset_path, cache)
        return dataset

    def get_dataset(
            self,
            dataset: str,
            cache: bool = True,
            cache_max_age: int = 60,
            cache_dir: typing.Union[None, str] = None,
            features: typing.Union[None, list] = None,
            labels: typing.Union[None, list] = None,
            data_type="numpy",
            shuffle=False,
            train_ratio: typing.Union[int, float] = 1.0,
            test_ratio: int = None,
            return_type: typing.Union[typing.Type[tuple], typing.Type[dict]] = tuple
    ):
        all_releases = self._get_releases(cache=cache, cache_max_age=cache_max_age, cache_dir=cache_dir)
        latest_release = self._get_latest_release(all_releases)

        dataset_dir = self._download_dataset(latest_release_metadata=latest_release, cache_dir=cache_dir)
        dataset_names = [x.name.lower() for x in dataset_dir.glob("*") if x.is_dir()]

        if not dataset_names:
            raise RuntimeError(
                f"Could not find dataset with the name {dataset.lower()}. Available: {dataset_names}")

        dataset_path = dataset_dir / dataset

        metadata = self._get_metadata_file(dataset_path)

        dataset_df = self._extract_dataset(metadata, dataset_path, cache)

        dataset_df = self._preprocess_dataset(metadata, dataset_path, dataset_df)

        if shuffle:
            dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

        # Dataset feature/label split
        try:
            X = dataset_df[features] if features else dataset_df
        except KeyError as err:
            _LOGGER.error(f"Could not find feature keys '{features}'. Available: {list(dataset_df.columns)}")
            raise err

        try:
            Y = dataset_df[labels] if labels else None
        except KeyError as err:
            _LOGGER.error(f"Could not find label keys '{labels}'. Available: {dataset_df.columns}")
            raise err

        # Dataset ratio. If ratio is a float. we split by %. else by count.
        if isinstance(train_ratio, int):
            X_train = X[0:train_ratio]
            Y_train = Y[0:train_ratio]
            if test_ratio is None:
                test_end_index = len(X)
            else:
                test_end_index = train_ratio + test_ratio

            X_test = X[train_ratio:test_end_index]
            Y_test = Y[train_ratio:test_end_index]

        elif isinstance(train_ratio, float):
            train_size = int(X.shape[0]*train_ratio)
            X_train = X[0:train_size]
            X_test = X[train_size:]
            Y_train = Y[0:train_size]
            Y_test = Y[train_size:]
        else:
            raise RuntimeError("train_ratio is not defined as int or float.")

        # Dataset type conversion
        if data_type == "numpy":
            X_train = X_train.values
            X_test = X_test.values
            Y_train = Y_train.values.flatten()
            Y_test = Y_test.values.flatten()

        if return_type == tuple:
            return X_train, Y_train, X_test, Y_test
        elif return_type == dict:
            return dict(
                x_train=X_train,
                y_train=Y_train,
                x_test=X_test,
                y_test=Y_test
            )
        else:
            raise RuntimeError("Invalid return_type. Should be set as 'dict' or 'tuple'")

    def list_datasets(self, cache=True, cache_max_age=60):
        all_releases = self._get_releases(cache=cache, cache_max_age=cache_max_age)
        latest_release = self._get_latest_release(all_releases)
        dataset_dir = self._download_dataset(latest_release_metadata=latest_release)

        return [x.name for x in dataset_dir.glob("*") if x.is_dir()]


class TMUDataset:

    def __init__(self):
        pass

    @abc.abstractmethod
    def _transform(self, name, dataset):
        raise NotImplementedError("You should override def _transform()")

    @abc.abstractmethod
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("You should override def _retrieve_dataset()")

    def get(self):
        return {k: self._transform(k, v) for k, v in self._retrieve_dataset().items()}

    def get_list(self):
        return list(self.get().values())


class MNIST(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        kwargs = dict()
        pyver = tuple([int(x) for x in sklearn.__version__.split(".")])

        if pyver[0] >= 1 and pyver[1] >= 2:
            kwargs["parser"] = "pandas"

        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            **kwargs
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=10000)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        return dict(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test
        )

    def _transform(self, name, dataset):
        if name.startswith("y"):
            return dataset

        return np.where(dataset.reshape((dataset.shape[0], 28 * 28)) > 75, 1, 0)


if __name__ == "__main__":

    MNIST().get()

    class TestBase(TMUDatasetSource):

        def __init__(self):
            pass


    data = TestBase().get_dataset(
        "XOR_biased",
        cache=False,
        cache_max_age=1,
        features=["X", "Y"],
        labels=["xor"],
        shuffle=True,
        train_ratio=1000
    )

    print(data[0].shape, data[1].shape)
