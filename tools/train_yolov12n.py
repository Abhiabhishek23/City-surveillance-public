#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Indian_Urban_Dataset_yolo' / 'data.yaml'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolov12n.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='')  # '' = auto
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(MODELS_DIR),
        name='yolov12n_indian_urban',
        exist_ok=True
    )
    print(result)

if __name__ == '__main__':
    main()
