"""Start an independent benchmark task session in an existing model process."""
import argparse
import json
from pathlib import Path


def main():
    import msgpack_numpy as mnp
    import zmq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.setsockopt(zmq.SNDTIMEO, 10000)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f'tcp://127.0.0.1:{args.port}')
        socket.send(mnp.packb({'endpoint': 'begin_evaluation'}))
        if mnp.unpackb(socket.recv(), raw=False) is not True:
            raise RuntimeError('Failed to start independent evaluation session')
        socket.send(mnp.packb({'endpoint': 'get_modality_config'}))
        info = mnp.unpackb(socket.recv(), raw=False)
        if (info.get('action_contract') != 'holomotion_reference_v1'
                or info.get('execution_steps') != 16 or info.get('reference_hz') != 50):
            raise ValueError('Unexpected model/controller contract')
        args.output.write_text(json.dumps(info, indent=2))
    finally:
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
