import os
import importlib
from os import path as osp


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def import_registered_modules(registration_folder="registrations"):
    """
    Import all registered modules from the specified folder.

    This function automatically scans all the files under the specified folder and imports all the required modules for registry.

    Parameters:
        registration_folder (str, optional): Path to the folder containing registration modules. Default is "registrations".

    Returns:
        list: List of imported modules.
    """

    print("\n")

    registration_modules_folder = (
        osp.dirname(osp.abspath(__file__)) + f"/{registration_folder}"
    )
    print("registration_modules_folder = ", registration_modules_folder)

    registration_modules_file_names = [
        osp.splitext(osp.basename(v))[0]
        for v in scandir(dir_path=registration_modules_folder)
    ]
    print("registration_modules_file_names = ", registration_modules_file_names)

    imported_modules = [
        importlib.import_module(f"{registration_folder}.{file_name}")
        for file_name in registration_modules_file_names
    ]
    print("imported_modules = ", imported_modules)
    print("\n")
