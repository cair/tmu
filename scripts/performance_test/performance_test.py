import os
import pathlib
import json
import sys
import subprocess
from git import Repo
import argparse


def run_process(cmd):
    lines = []
    # Start the subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Print the output of the subprocess as it runs
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line.decode().strip())
        lines.append(line.decode().strip())

    # Wait for the subprocess to finish and get its return code
    returncode = proc.wait()

    return lines, returncode


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--git-url", default="https://github.com/cair/tmu.git", type=str)
    parser.add_argument("--branch", default="dev", type=str)
    parser.add_argument("--tests", nargs='+', default=["MNISTDemo.py"])
    args = parser.parse_args()

    current_dir = pathlib.Path(__file__).parent
    repo_dir = current_dir.joinpath("repo")
    tmu_dir = repo_dir
    tmu_examples_dir = tmu_dir / "examples"
    match_config = json.load(current_dir.joinpath("match.json").open())
    venv_vin = current_dir / "venv" / "bin"
    python_executable = venv_vin / "python"
    pip_executable = venv_vin / "pip"

    # Create virtual env
    os.chdir(current_dir)
    os.system("python -m venv venv")

    if repo_dir.exists():
        repo = Repo(repo_dir)
    else:
        repo = Repo.clone_from(args.git_url, repo_dir)

    # Checkout branch
    repo.git.checkout(args.branch)

    # Get all commits
    all_commits = list(repo.iter_commits())

    for commit in all_commits:
        repo.git.checkout(commit)
        # repo.git.reset('--hard', commit)

        # Compile new tmulib
        curr_dir = os.curdir
        os.chdir(repo_dir)
        os.system(f"{pip_executable.absolute()} install -e .")
        os.system(f"{pip_executable.absolute()} install -r examples/requirements.txt")
        os.system(f"{pip_executable.absolute()} install tqdm")
        os.chdir(curr_dir)

        for test in args.tests:

            test_files = list(tmu_examples_dir.rglob(f"*{test}*"))
            if len(test_files) == 0:
                raise RuntimeError("Could not find file")
            test_file = test_files[0]

            print("------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------")
            print("------------------------------------------------------------------------------------")
            print(test_file, commit, commit.message)
            run_process([python_executable.absolute(), test_file, "--epochs=2"])
