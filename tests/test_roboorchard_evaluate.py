"""Regression checks for read-only inputs and evaluation failure reporting."""
import hashlib
import io
import json
import tarfile

import pytest

from locomanilab.runtime.roboorchard.assets import import_assets
from locomanilab.runtime.roboorchard.evaluate import local_output, validate_result


def test_bucket_output_alias_is_rejected(tmp_path):
    (tmp_path / 'alias').symlink_to('/horizon-bucket')
    with pytest.raises(ValueError, match='read-only bucket'):
        local_output('alias/results', tmp_path)


def test_request_errors_cannot_be_reported_as_completed(tmp_path):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps(dict(episodes=1, errors=1, results=[{}])))
    (tmp_path / 'episode.mp4').write_bytes(b'video')
    with pytest.raises(RuntimeError, match='Incomplete'):
        validate_result(path, 1)
    path.write_text(json.dumps(dict(episodes=1, errors=0, results=[{'success': False}])))
    assert validate_result(path, 1)['results'][0]['success'] is False
    (tmp_path / 'episode.mp4').write_bytes(b'')
    with pytest.raises(RuntimeError, match='video'):
        validate_result(path, 1)


def bundle(tmp_path, name, content=b'asset'):
    archive = tmp_path / 'assets.tar.gz'
    record = dict(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    manifest = dict(git_files={}, files={name: record}, repository='example', revision='abc', simulator_revision='def')
    with tarfile.open(archive, 'w:gz') as tar:
        for filename, data in [('manifest.json', json.dumps(manifest).encode()), ('assets/' + name, content)]:
            entry = tarfile.TarInfo(filename)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    archive.with_name(archive.name + '.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive


def test_asset_import_and_conflict_preservation(tmp_path):
    archive = bundle(tmp_path, 'object/model.xml')
    dest = tmp_path / 'private'
    assert import_assets(archive, dest)['revision'] == 'abc'
    target = dest / 'object/model.xml'
    assert target.read_bytes() == b'asset'
    target.write_bytes(b'user data')
    with pytest.raises(ValueError, match='differs'):
        import_assets(archive, dest)
    assert target.read_bytes() == b'user data'


def test_asset_traversal_is_rejected(tmp_path):
    archive = bundle(tmp_path, '../escape')
    with pytest.raises(ValueError, match='escapes'):
        import_assets(archive, tmp_path / 'private')
    assert not (tmp_path / 'escape').exists()
