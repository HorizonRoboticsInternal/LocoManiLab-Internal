"""Run a task-based, offline RoboOrchard evaluation inside a provisioned worker.

This module has no AIDI dependency. The submit-side package generator supplies
source snapshots and explicit local/read-only input paths.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from locomanilab.benchmark import load_tasks
from locomanilab.runtime import script_path
from locomanilab.runtime.roboorchard.assets import import_assets

SCHEMA = 'locomanilab.roboorchard.eval.v1'
INPUTS = ('roboorchard_repo', 'benchmark_repo', 'g1_backend_repo', 'job', 'policy_config', 'assets_archive')


def resolve(value, base):
    expanded = os.path.expandvars(str(value))
    if '$' in expanded:
        raise ValueError(f'Unresolved environment variable: {value}')
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def local_output(value, base):
    path = resolve(value, base)
    if path.is_relative_to('/horizon-bucket'):
        raise ValueError('Output/runtime must be outside the read-only bucket')
    return path


def load_config(path):
    """Resolve and validate without starting models, creating files or networking."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get('schema') != SCHEMA:
        raise ValueError(f'Expected schema {SCHEMA}')
    for key in INPUTS:
        cfg[key] = str(resolve(cfg[key], path.parent))
        if not Path(cfg[key]).exists():
            raise ValueError(f'Missing {key}: {cfg[key]}')
    cfg['output'] = str(local_output(cfg['output'], path.parent))
    for key in INPUTS:
        source = Path(cfg[key])
        if Path(cfg['output']).is_relative_to(source) or source.is_relative_to(cfg['output']):
            raise ValueError(f'Output overlaps read-only input {key}')
    if not isinstance(cfg.get('checkpoint'), str) or cfg['checkpoint'] in ('', '.', '..') or Path(cfg['checkpoint']).name != cfg['checkpoint']:
        raise ValueError('Select one explicit checkpoint name (for example checkpoint_0)')
    if not (Path(cfg['job']) / 'checkpoints' / cfg['checkpoint']).is_dir():
        raise ValueError('Selected checkpoint is missing')
    import yaml
    policy = yaml.safe_load(Path(cfg['policy_config']).read_text())
    policy['models_dir'] = str(resolve(policy['models_dir'], Path(cfg['policy_config']).parent))
    if not Path(policy['models_dir']).is_dir():
        raise ValueError('Missing controller model directory')
    if Path(cfg['output']).is_relative_to(policy['models_dir']) or Path(policy['models_dir']).is_relative_to(cfg['output']):
        raise ValueError('Output overlaps controller models')
    cfg['resolved_policy'] = policy
    tasks = load_tasks()
    names = cfg.get('tasks')
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in tasks for n in names):
        raise ValueError('tasks must be a nonempty list of registered benchmark task names')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate task names')
    for key, default in (('cpu_limit', 4), ('startup_timeout', 900), ('task_timeout', 1800)):
        cfg.setdefault(key, default)
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['cpu_limit'] > 8:
        raise ValueError('cpu_limit must be <= 8 for this single-worker recipe')
    for key in ('episodes', 'max_steps'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    cfg.setdefault('seed', 0)
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    for role in ('model', 'sim'):
        cfg.setdefault(role + '_python', sys.executable)
        paths = cfg.setdefault(role + '_python_path', [])
        if not isinstance(paths, list):
            raise ValueError(f'{role}_python_path must be a list')
        cfg[role + '_python_path'] = [str(resolve(p, path.parent)) for p in paths]
        if any(Path(cfg['output']).is_relative_to(p) or Path(p).is_relative_to(cfg['output'])
               for p in cfg[role + '_python_path']):
            raise ValueError('Output overlaps dependency overlay')
        if any(not Path(p).is_dir() for p in cfg[role + '_python_path']):
            raise ValueError(f'Missing {role} dependency overlay')
    if not Path(cfg['assets_archive'] + '.sha256').is_file():
        raise ValueError('Asset archive requires its .sha256 sidecar')
    return cfg


def child_env(cfg, role, runtime, simulator):
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[3])
    paths = list(cfg[role + '_python_path'])
    if role == 'sim':
        paths += [str(simulator), str(Path(cfg['benchmark_repo']) / 'robosuite_src'), cfg['g1_backend_repo']]
    paths += [package_root]
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE='1',
               HOME=str(runtime / 'home'), TMPDIR=str(runtime / 'temp'),
               HF_HOME=str(runtime / 'hf'), HF_HUB_OFFLINE='1',
               XDG_CACHE_HOME=str(runtime / 'cache'), NUMBA_CACHE_DIR=str(runtime / 'numba'),
               TORCH_EXTENSIONS_DIR=str(runtime / 'torch_extensions'),
               TRITON_CACHE_DIR=str(runtime / 'triton'), CUDA_CACHE_PATH=str(runtime / 'cuda'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0')
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, [env.get(key), 'localhost', '127.0.0.1']))
    return env


def stop(process):
    if process is None:
        return
    # A native server is a child of the bridge; terminate the entire owned group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    # The group leader may have exited before its native descendants.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def task_command(cfg, task, port, output):
    cmd = [cfg['sim_python'], str(script_path('sim_client.py')),
           '--backend', 'holomotion', '--robot', 'G1Revo2',
           '--policy-config', cfg['policy_config'], '--inference-backend', 'onnx-cpu',
           '--control-freq', '50', '--inference-rate', '3.125', '--host', '127.0.0.1', '--port', str(port),
           '--env', task.env, '--scene-id', str(task.scene_id), '--prompt', task.prompt,
           '--episodes', str(cfg.get('episodes', task.episodes)), '--max-steps', str(cfg.get('max_steps', task.max_steps)),
           '--genuine-min-steps', str(task.genuine_min_steps), '--fixed-scene',
           '--scene-seed', str(task.scene_seed), '--reset-jitter', str(task.reset_jitter),
           '--reset-object-jitter', str(task.reset_object_jitter), '--jitter-seed', str(task.jitter_seed),
           '--seed', str(cfg['seed']), '--render', 'video', '--video-dir', str(output),
           '--results-json', str(output / 'results.json')]
    return cmd


def validate_result(path, episodes):
    result = json.loads(Path(path).read_text())
    rows = result.get('results', [])
    if result.get('episodes') != episodes or len(rows) != episodes or result.get('errors', 0) or any(r.get('error') for r in rows):
        raise RuntimeError('Incomplete evaluation or model-request errors; inspect task log')
    videos = list(Path(path).parent.glob('*.mp4'))
    if len(videos) != episodes or any(p.stat().st_size == 0 for p in videos):
        raise RuntimeError('Missing episode video; inspect task log')
    return result


def run(cfg):
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    import yaml
    (output / 'policy.yaml').write_text(yaml.safe_dump(cfg['resolved_policy']))
    cfg = dict(cfg, policy_config=str(output / 'policy.yaml'))
    runtime = output / 'runtime'
    for name in ('home', 'temp', 'cache', 'numba', 'torch_extensions', 'triton', 'cuda'):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    (output / 'resolved_config.json').write_text(json.dumps(cfg, indent=2))
    summary = {'status': 'preparing', 'tasks': [], 'checkpoint': cfg['checkpoint']}
    model = sim = None
    previous = {}
    def interrupted(signum, frame):
        raise RuntimeError(f'Evaluation interrupted by signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, interrupted)
    try:
        if hasattr(os, 'sched_getaffinity'):
            allowed = sorted(os.sched_getaffinity(0))[:cfg['cpu_limit']]
            os.sched_setaffinity(0, allowed)
            summary['cpu_affinity'] = allowed
        # No symlinks back to the read-only asset source: simulator writes derived XML beside assets.
        simulator = runtime / 'simulator'
        shutil.copytree(Path(cfg['benchmark_repo']) / 'robocasa', simulator / 'robocasa',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        summary['assets'] = import_assets(cfg['assets_archive'], simulator / 'robocasa/models/assets')
        model_env = child_env(cfg, 'model', runtime, simulator)
        sim_env = child_env(cfg, 'sim', runtime, simulator)
        for role, env in (('model', model_env), ('sim', sim_env)):
            code = ('import torch, transformers, zmq, msgpack_numpy; '
                    'assert torch.cuda.is_available(), "CUDA unavailable"; '
                    'print(torch.__version__, transformers.__version__)' if role == 'model' else
                    'import mujoco, numpy, numba, onnxruntime, omegaconf, robocasa, robosuite; '
                    'assert mujoco.__version__ == "3.3.1"; print(mujoco.__version__, numpy.__version__)')
            with (output / f'{role}_preflight.log').open('w') as log:
                subprocess.run([cfg[role + '_python'], '-c', code], env=env, cwd=runtime,
                               stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [cfg['model_python'], str(script_path('roboorchard/server.py')),
                   '--repo', cfg['roboorchard_repo'], '--job', cfg['job'], '--checkpoint', cfg['checkpoint'],
                   '--runtime-root', str(runtime / 'model'), '--port', str(port), '--seed', str(cfg['seed'])]
        with (output / 'model.log').open('w') as log:
            model = subprocess.Popen(command, env=model_env, cwd=runtime, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + cfg['startup_timeout']
        while True:
            if model.poll() is not None:
                raise RuntimeError('Model server exited before readiness; inspect model.log')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Model startup timeout')
                time.sleep(.5)
        for name in cfg['tasks']:
            task = load_tasks()[name]
            target = output / name
            target.mkdir()
            subprocess.run([cfg['model_python'], str(script_path('roboorchard/session.py')),
                            '--port', str(port), '--output', str(target / 'model_info.json')],
                           env=model_env, cwd=runtime, check=True, timeout=30)
            command = task_command(cfg, task, port, target)
            (target / 'command.json').write_text(json.dumps(command, indent=2))
            with (target / 'run.log').open('w') as log:
                sim = subprocess.Popen(command, env=sim_env, cwd=runtime, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + cfg['task_timeout']
                while sim.poll() is None:
                    if model.poll() is not None:
                        raise RuntimeError('Model server exited during evaluation')
                    if time.monotonic() > deadline:
                        raise TimeoutError(f'{name} exceeded task_timeout')
                    time.sleep(.5)
                code = sim.returncode
            if code:
                raise RuntimeError(f'{name} simulator exited {code}')
            result = validate_result(target / 'results.json', cfg.get('episodes', task.episodes))
            summary['tasks'].append({'task': name, 'results': result})
            (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        summary['status'] = 'completed'
    except Exception as exc:
        summary.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        cleanup_errors = []
        for process in (sim, model):
            try:
                stop(process)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        summary['resources_released'] = not cleanup_errors
        if cleanup_errors:
            summary.update(status='error', cleanup_errors=cleanup_errors)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if summary.get('cleanup_errors'):
        raise RuntimeError('Process cleanup failed; inspect summary.json')
    print(json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Validate paths/config only, without writes or model execution')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.check:
        print(json.dumps(cfg, indent=2))
    else:
        run(cfg)


if __name__ == '__main__':
    main()
