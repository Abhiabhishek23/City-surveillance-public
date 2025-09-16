"""CCTV client wrapper

Runs on a more powerful machine. Uses YOLOv12-s by default and posts alerts
to the backend /alerts endpoint when overcrowding or illegal construction
is detected by the shared monitoring logic.
"""

import argparse
import os

# Import the shared client module (the heavy imports only happen at runtime)
import urban_monitor_deploy as um


def run(model='yolov12-s', source='0', backend='http://127.0.0.1:8000/alerts', camera='CCTV01', enable_verifier=False):
    # Resolve model using same helper; allow alias like 'yolov12-s'
    model_path = um.resolve_model(model)
    um.main(model_path, source, backend, camera, enable_verifier=enable_verifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCTV AI Client (YOLOv12-s)')
    parser.add_argument('--model', type=str, default='yolov12-s', help='Model path or alias (default: yolov12-s)')
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file path')
    parser.add_argument('--backend', type=str, default='http://127.0.0.1:8000/alerts', help='Backend alerts URL')
    parser.add_argument('--camera', type=str, default='CCTV01', help='Camera id')
    parser.add_argument('--enable-verifier', action='store_true', help='Enable Mask R-CNN verifier (requires torchvision)')
    args = parser.parse_args()

    run(model=args.model, source=args.source, backend=args.backend, camera=args.camera, enable_verifier=args.enable_verifier)
