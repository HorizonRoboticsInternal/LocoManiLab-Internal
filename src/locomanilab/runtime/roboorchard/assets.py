"""Verify an offline task asset bundle and populate a private simulator copy."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def import_assets(archive, destination):
    """Import verified regular files; never overwrite a conflicting source asset."""
    archive, destination = Path(archive), Path(destination).resolve()
    checksum = archive.with_name(archive.name + '.sha256')
    expected = checksum.read_text().split()[0]
    if sha256(archive) != expected:
        raise ValueError('Asset archive checksum mismatch')
    with tarfile.open(archive) as stream:
        manifest = json.load(stream.extractfile('manifest.json'))
        for name, record in {**manifest['git_files'], **manifest['files']}.items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Asset path escapes destination: {name}')
            if target.exists():
                if not target.is_file() or sha256(target) != record['sha256']:
                    raise ValueError(f'Existing asset differs: {name}')
            elif name in manifest['git_files']:
                raise ValueError(f'Missing pinned source asset: {name}')
        remaining = set(manifest['files'])
        for member in stream:
            if not member.name.startswith('assets/'):
                continue
            name = member.name[len('assets/'):]
            if name not in remaining or not member.isfile():
                raise ValueError(f'Unexpected or duplicate asset entry: {name}')
            record = manifest['files'][name]
            target = destination / name
            if member.size != record['size']:
                raise ValueError(f'Asset size mismatch: {name}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.importing')
            try:
                with stream.extractfile(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if sha256(temporary) != record['sha256']:
                    raise ValueError(f'Asset hash mismatch: {name}')
                if target.exists():
                    temporary.unlink()
                else:
                    temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            remaining.remove(name)
        if remaining:
            raise ValueError(f'Archive missing {len(remaining)} assets')
    return {key: manifest[key] for key in ('repository', 'revision', 'simulator_revision')}
