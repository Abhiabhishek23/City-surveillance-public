"""Drone client wrapper

Runs on the drone's onboard computer. Uses a lightweight Edge YOLO model by default
for real-time processing. Posts alerts to the backend /alerts endpoint when events
are detected.
"""

import argparse

import urban_monitor_deploy as um


def run(model='edge-yolo', source='0', backend='http://127.0.0.1:8000/alerts', camera='DRONE01'):
    # For edge models, user should supply a path or alias that resolve_model can find.
    model_path = um.resolve_model(model)
    um.main(model_path, source, backend, camera, enable_verifier=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drone AI Client (Edge YOLO)')
    parser.add_argument('--model', type=str, default='edge-yolo', help='Model path or alias for edge model')
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file path')
    parser.add_argument('--backend', type=str, default='http://127.0.0.1:8000/alerts', help='Backend alerts URL')
    parser.add_argument('--camera', type=str, default='DRONE01', help='Camera id')
    args = parser.parse_args()

    run(model=args.model, source=args.source, backend=args.backend, camera=args.camera)
