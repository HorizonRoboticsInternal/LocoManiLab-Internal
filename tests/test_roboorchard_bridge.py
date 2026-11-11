"""Round-trip the HTTP payload and model reset through the compatibility bridge."""
import json
from contextlib import contextmanager
import numpy as np
from locomanilab.runtime.roboorchard.server import PolicyBridge, encode, decode


def test_bridge_preserves_state_image_and_episode_identity(monkeypatch):
    requests = []
    @contextmanager
    def urlopen(request, timeout):
        payload = json.loads(request.data, object_hook=decode)
        requests.append(payload)
        action = np.zeros((50,87), np.float32)
        action[:,61] = 1
        yield type('Reply', (), {'read': lambda self: json.dumps({'action': action}, default=encode).encode()})()
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    bridge = PolicyBridge('http://127.0.0.1:1')
    image = np.arange(18, dtype=np.uint8).reshape(1,1,2,3,3)
    joints = np.arange(51, dtype=np.float32)
    obs = dict(video={'ego_view_left': image}, state={'joints': joints[None,None],
               'base_pose': np.array([1,2,.8,1,0,0,0])[None,None]},
               language={'annotation.human.task_description': [['pick the bottle']]})
    assert bridge.get_action(obs)['reference'].shape == (50,87)
    bridge.get_action(obs)
    bridge.reset()
    bridge.get_action(obs)
    assert [p['history']['step_index'] for p in requests] == [0,16,0]
    assert [p['history']['reset'] for p in requests] == [True,False,True]
    assert requests[2]['history']['episode_index'] == requests[0]['history']['episode_index'] + 1
    np.testing.assert_array_equal(requests[0]['image']['observation.images.ego_view_left'], image[0,0])
    np.testing.assert_array_equal(requests[0]['state']['observation.state'], joints)


def test_model_server_uses_saved_job_and_model_environment(ctx, tmp_path):
    from locomanilab.registry import build_plugin
    from locomanilab.plugin import SimServer, StageResult
    repo = tmp_path / 'model repo'
    repo.mkdir()
    ctx.repos['roboorchard'] = str(repo)
    plugin = build_plugin('roboorchard', ctx, dry_run=True)
    missing = plugin._sim_server(stage='eval', prompt='pick', port=5550)
    assert isinstance(missing, StageResult) and not missing.ok
    server = plugin._sim_server(stage='eval', prompt='pick', port=5550, run_dir='saved-job')
    assert isinstance(server, SimServer)
    assert server.argv[:2] == ['bash', '-lc']
    assert 'exec uv run --no-sync python' in server.argv[2]
    assert '--job ' in server.argv[2] and '--checkpoint ' not in server.argv[2]
    assert server.cwd == repo and server.inference_rate == 3.125
    assert not plugin._sim_server(stage='eval', backend='sonic', prompt='pick', port=5550).ok


def test_saved_archive_is_materialized_without_modifying_job(tmp_path):
    import io
    import tarfile
    from locomanilab.runtime.roboorchard.server import saved_source
    job = tmp_path / 'read-only-job'
    job.mkdir()
    archive = job / 'training_source.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name, data in (('.training_source_complete', b'ok'), ('robo_orchard_lab/__init__.py', b'# saved source')):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    original = archive.read_bytes()
    source = saved_source(job, tmp_path / 'runtime')
    assert (source / 'robo_orchard_lab/__init__.py').read_bytes() == b'# saved source'
    assert saved_source(job, tmp_path / 'runtime') == source
    assert list(job.iterdir()) == [archive] and archive.read_bytes() == original
