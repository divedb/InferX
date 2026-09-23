"""Provenance capture for reproducible runs: digests, environment, sources.

Every run directory records enough metadata to answer "what exactly was
measured": git head and dirty diff, a source archive, checkpoint/config/
workload hashes, harness hashes, binary hash, GPU/driver, package versions,
and the exact commands used. The offline runner additionally verifies the
InferX binary did not change during a run.
"""
import hashlib
import importlib.metadata
import subprocess
import tarfile
from pathlib import Path


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   text=True, cwd=root).strip()


def platform() -> str:
    return subprocess.check_output(['uname', '-a'], text=True).strip()


def package_versions(names) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'absent'
    return versions


def harness_hashes(directories) -> dict:
    """SHA256 of every Python file in the given benchmark directories."""
    hashes = {}
    for directory in directories:
        for path in sorted(Path(directory).glob('*.py')):
            hashes[str(path)] = digest(path)
    return hashes


def write_source_archive(root: Path, out: Path, extra_py_dirs=()) -> None:
    """Snapshot engine sources plus the benchmark harness into out.tar.gz."""
    with tarfile.open(out, 'w:gz') as archive:
        for folder in ('src', 'include', 'apps', 'tests'):
            for path in sorted((root / folder).rglob('*')):
                if path.is_file() and path.suffix in ('.h', '.cc', '.cu', '.txt', '.cmake'):
                    archive.add(path)
        archive.add(root / 'CMakeLists.txt')
        for directory in extra_py_dirs:
            for path in sorted(Path(directory).glob('*.py')):
                archive.add(path)
