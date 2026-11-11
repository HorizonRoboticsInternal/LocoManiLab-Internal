"""Expose the native RoboOrchard HoloMotion service over the shared ZMQ API.

Runs in the model environment. The native server retains checkpoint/source,
tokenizer, image preprocessing and action decoding ownership.
"""
from __future__ import annotations

import argparse
import base64
import json
import hashlib
import tarfile
import tempfile
import shutil
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

import numpy as np


def encode(value):
    if isinstance(value, (np.ndarray, np.generic)):
        a = np.asarray(value)
        return {'__numpy__': base64.b64encode(a.tobytes()).decode(), 'dtype': a.dtype.str, 'shape': a.shape}
    raise TypeError(type(value).__name__)


def decode(value):
    if '__numpy__' in value:
        return np.frombuffer(base64.b64decode(value['__numpy__']), dtype=value['dtype']).reshape(value['shape'])
    return value


class PolicyBridge:
    def __init__(self, url):
        self.url = url
        self.session = str(uuid.uuid4())
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.step = 0
        return True

    def get_action(self, obs):
        state = obs['state']
        image = np.asarray(obs['video']['ego_view_left'])
        if image.ndim != 5 or image.shape[:2] != (1, 1) or image.dtype != np.uint8 or image.shape[-1] != 3:
            raise ValueError('Expected uint8 video.ego_view_left[1,1,H,W,3]')
        joints = np.asarray(state['joints']).reshape(-1)
        pose = np.asarray(state['base_pose']).reshape(-1)
        if joints.shape != (51,) or pose.shape != (7,) or not np.isfinite(np.r_[joints, pose]).all():
            raise ValueError('Expected finite Revo2 joints[51] and base_pose[7]')
        instruction = obs['language']['annotation.human.task_description'][0][0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Expected a nonempty instruction')
        payload = dict(
            image={'observation.images.ego_view_left': image[0, 0]},
            instruction=instruction,
            state={'observation.state': joints, 'observation.base_pose': pose},
            history=dict(session_id=self.session, episode_index=self.episode, step_index=self.step,
                         reset=self.step == 0),
            dataset_name='holomotion_revo2',
        )
        request = urllib.request.Request(self.url + '/act', json.dumps(payload, default=encode).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read(), object_hook=decode)
        action = np.asarray(result['action'], dtype=np.float32)
        if action.shape != (50, 87) or not np.isfinite(action).all():
            raise ValueError('Native server must return finite reference[50,87]')
        self.step += 16
        return {'reference': action}



def saved_source(job, runtime_root):
    """Materialize a saved archive locally without writing to the saved job."""
    archive = job / 'training_source.tar.gz'
    if not archive.is_file():
        return None  # Native server validates the legacy in-job snapshot.
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    parent = runtime_root / 'sources'
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / digest.hexdigest()
    if (destination / '.training_source_complete').is_file():
        return destination
    staging = Path(tempfile.mkdtemp(prefix='extract-', dir=parent))
    try:
        with tarfile.open(archive, 'r:gz') as stream:
            members = stream.getmembers()
            for member in members:
                target = (staging / member.name).resolve()
                if not target.is_relative_to(staging.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError(f'Unsafe saved source member: {member.name}')
            stream.extractall(staging, members=members)
        if not (staging / '.training_source_complete').is_file():
            raise ValueError('Saved training source archive is incomplete')
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def main():
    import msgpack_numpy as mnp
    import zmq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5550)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    native = args.repo / 'projects/holobrain_internal/humanoid/scripts/serve_qwen3_151d_holomotion.py'
    if not native.is_file():
        raise FileNotFoundError(native)
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1', 0))
        native_port = port_socket.getsockname()[1]
    cmd = [sys.executable, str(native), '--job', str(args.job), '--runtime-root', str(args.runtime_root), '--host', '127.0.0.1', '--port', str(native_port)]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed), '--request-seeding'])
    source = saved_source(args.job, args.runtime_root)
    if source is not None:
        cmd.extend(['--source-root', str(source)])
    proc = subprocess.Popen(cmd, cwd=args.repo)
    def stop(_signal, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    url = f'http://127.0.0.1:{native_port}'
    sock = None
    try:
        deadline = time.monotonic() + 850
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'RoboOrchard server exited with {proc.returncode}')
            try:
                with urllib.request.urlopen(url + '/info', timeout=2) as response:
                    info = json.loads(response.read())
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('RoboOrchard startup timed out')
                time.sleep(.25)
        if info.get('action_contract') != 'holomotion_unified_151d_v1' or info.get('response_action_dim') != 87:
            raise ValueError('Native server does not advertise the unified-v1 Revo2 contract')
        policy = PolicyBridge(url)
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.bind(f'tcp://127.0.0.1:{args.port}')
        while True:
            request = mnp.unpackb(sock.recv(), raw=False)
            endpoint = request.get('endpoint')
            try:
                if endpoint == 'get_action':
                    reply = [policy.get_action(request['data']['observation']), {}]
                elif endpoint == 'begin_evaluation':
                    policy = PolicyBridge(url)
                    reply = True
                elif endpoint == 'reset':
                    reply = policy.reset()
                elif endpoint == 'ping':
                    reply = proc.poll() is None
                elif endpoint == 'get_modality_config':
                    reply = {'controller': 'holomotion_v14', 'action_contract': 'holomotion_reference_v1',
                             'reference_hz': 50, 'execution_steps': 16, 'future_frames': 10, 'info': info}
                elif endpoint == 'kill':
                    sock.send(mnp.packb(True))
                    break
                else:
                    raise ValueError(f'Unknown endpoint {endpoint}')
                sock.send(mnp.packb(reply))
            except Exception as exc:
                print(f'RoboOrchard request failed: {exc}', file=sys.stderr, flush=True)
                sock.send(b'ERROR')
    finally:
        if sock is not None:
            sock.close(linger=0)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    main()
