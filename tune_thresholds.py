"""tune_thresholds.py
Sweep confidence and IoU thresholds for a single video using ultralytics YOLO model.
Saves annotated outputs under runs/tune/<name> and writes a CSV summary.
"""
import os
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO


def count_and_avg_conf(res):
    total = 0
    confs = []
    try:
        boxes = res.boxes
        # Boxes may expose xyxy or be list-like
        if hasattr(boxes, 'xyxy'):
            total = len(boxes.xyxy)
        else:
            try:
                total = len(boxes)
            except Exception:
                total = 0
        # confidences
        if hasattr(boxes, 'conf'):
            try:
                confs = boxes.conf.cpu().numpy().tolist()
            except Exception:
                try:
                    confs = [float(c) for c in boxes.conf]
                except Exception:
                    confs = []
    except Exception:
        return 0, 0.0
    avg = float(sum(confs) / len(confs)) if confs else 0.0
    return total, avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default='Test samples/3.mp4', help='Path to video to tune')
    p.add_argument('--confs', nargs='+', type=float, default=[0.3,0.4,0.5], help='Confidence thresholds')
    p.add_argument('--ious', nargs='+', type=float, default=[0.3,0.5], help='IOU thresholds')
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    video = args.video
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        return

    MODEL_PATH = os.getenv('YOLO_MODEL', 'yolov8n.pt')
    print('Using model:', MODEL_PATH)
    model = YOLO(MODEL_PATH)

    out_dir_base = Path('runs/tune')
    out_dir_base.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir_base / 'summary.csv'
    first_write = not summary_path.exists()

    rows = []
    video_id = Path(video).name
    for conf in args.confs:
        for iou in args.ious:
            name = f'conf{conf}_iou{iou}'.replace('.', '_')
            project = str(out_dir_base)
            print(f'Running conf={conf} iou={iou} -> project={project} name={name}')
            try:
                results = model.predict(source=video, conf=conf, iou=iou, imgsz=args.imgsz, save=True, project=project, name=name)
            except Exception as e:
                print('Inference failed for', conf, iou, 'error:', e)
                continue

            # results is a list (frames or batch results) or a Results object
            total_dets = 0
            avg_conf = 0.0
            per_frame = 0
            try:
                # If results is iterable of frame Results
                for res in results:
                    per_frame += 1
                    n, a = count_and_avg_conf(res)
                    total_dets += n
                    # weighted average
                    avg_conf = (avg_conf * (per_frame-1) + a) / per_frame if per_frame else a
            except Exception:
                # single result
                try:
                    n, a = count_and_avg_conf(results)
                    total_dets = n
                    avg_conf = a
                    per_frame = 1
                except Exception:
                    total_dets = 0
                    avg_conf = 0.0
                    per_frame = 0

            out_path = out_dir_base / name
            rows.append((conf, iou, per_frame, total_dets, round(avg_conf,4), str(out_path), video_id))
            print(f'-> frames={per_frame} total_detections={total_dets} avg_conf={avg_conf:.4f} saved to {out_path}')

    # write summary
    write_header = first_write
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['conf','iou','frames','total_detections','avg_conf','output_dir','video'])
        for r in rows:
            writer.writerow(r)

    print('Summary written to', summary_path)

if __name__ == '__main__':
    main()
