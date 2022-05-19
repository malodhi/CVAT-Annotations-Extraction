import sys
import subprocess as sp
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].as_posix())

from typing import Union, List
import os
from natsort import natsorted


def collect_file_paths(
        path: str, suffixes: Union[List[str], List[str]] = None,
        flatten: bool = True, sort: bool = True) -> list:
    """Returns list of file_paths with matching suffixes"""
    suffixes = [] if suffixes is None else suffixes
    collected_file_paths = list()
    if not path:
        return collected_file_paths

    try:
        search_paths = os.listdir(path)
    except NotADirectoryError:
        search_paths = [path]
    except FileNotFoundError:
        search_paths = []

    for search_path in search_paths:
        abs_search_path = os.path.join(path, search_path)
        # If search path is a directory, then,
        # collect file paths from it and append to collection
        if os.path.isdir(abs_search_path):
            file_paths = collect_file_paths(
                path=abs_search_path,
                suffixes=suffixes,
                flatten=flatten
            )
            collected_file_paths += file_paths if flatten or not file_paths else [file_paths]
        # If search path is a file, then,
        # append directly to collection
        else:
            file_name = Path(abs_search_path).name
            if suffixes and not file_name.endswith(tuple(suffixes)):
                collected_file_paths += []
            else:
                collected_file_paths += [abs_search_path]
    if sort:
        collected_file_paths = natsorted(collected_file_paths)
    return collected_file_paths


if __name__ == "__main__":
    src_dir_path = "/home/mansoor/Downloads/rotate_images/images"
    im_file_paths = collect_file_paths(src_dir_path, ["jpg", "jpeg", "png"])

    for im_file_path in im_file_paths:
        sp.run(["exiftool", "-all=", f"{im_file_path}", "-overwrite_original"])
        # sp.run(["convert", "-rotate", "+180", f"{im_file_path}", f"{im_file_path}"])
    # convert -rotate "-90" in.jpg out.jpg
