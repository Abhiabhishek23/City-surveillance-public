#!/usr/bin/env python3
import os
import io
import json
from typing import List, Dict

import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path
import subprocess
import sys
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title='Encroachment Annotator', layout='wide')

st.title('Encroachment Annotation UI')

# Sidebar config
images_dir = st.sidebar.text_input('Images directory', 'dataset/images/train')
output_jsonl = st.sidebar.text_input('Output JSONL', 'dataset/annotations_ui.jsonl')
labels_dir = st.sidebar.text_input('Output YOLO labels dir', 'dataset/labels/train')
data_yaml = st.sidebar.text_input('Top-class data.yaml path', 'dataset/data.yaml')
objects_data_yaml = st.sidebar.text_input('Objects data.yaml path', 'dataset/objects_data.yaml')
train_epochs = st.sidebar.number_input('Train epochs', min_value=1, max_value=200, value=20)
imgsz = st.sidebar.number_input('Image size', min_value=320, max_value=1280, value=640, step=32)
only_unlabeled = st.sidebar.checkbox('Only show unlabeled images', value=True)
csv_export_path = st.sidebar.text_input('Export CSV path', 'dataset/annotations_ui.csv')

# Attributes
permanence = st.sidebar.selectbox('Default permanence', ['temporary','permanent'], index=0)
permit_status = st.sidebar.selectbox('Default permit', ['unknown','approved','unapproved'], index=0)
zone = st.sidebar.selectbox('Default zone', ['none','river_buffer','road_footpath','vending_zone','festival_zone'], index=0)
area_type = st.sidebar.selectbox('Default area type', ['built','natural_area'], index=0)

# State
if 'boxes' not in st.session_state:
    st.session_state['boxes'] = []  # [{'bbox':[cx,cy,w,h], 'attributes':{...}}]
if 'current_image' not in st.session_state:
    st.session_state['current_image'] = None
if 'image_class' not in st.session_state:
    st.session_state['image_class'] = None
if 'img_idx' not in st.session_state:
    st.session_state['img_idx'] = 0
if 'label_mode' not in st.session_state:
    st.session_state['label_mode'] = 'Objects'
if 'object_class' not in st.session_state:
    st.session_state['object_class'] = 'vendor_cart_thela'

# Image list
def list_images(p: str) -> List[str]:
    files = []
    for root, _, names in os.walk(p):
        for n in names:
            if n.lower().endswith(('.jpg','.jpeg','.png')):
                files.append(os.path.join(root, n))
    return sorted(files)

def load_annotated_images(jsonl_path: str) -> set:
    annotated = set()
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    annotated.add(rec.get('image'))
        except Exception:
            pass
    return annotated

all_imgs = list_images(images_dir)
if only_unlabeled:
    annotated = load_annotated_images(output_jsonl)
    # Convert absolute paths to relative endings comparable to stored 'image'
    def to_rel(p: str) -> str:
        return p.split('dataset/')[-1] if 'dataset/' in p else p
    imgs = [p for p in all_imgs if to_rel(p) not in annotated]
else:
    imgs = all_imgs
if imgs:
    # Clamp index
    st.session_state['img_idx'] = max(0, min(st.session_state['img_idx'], len(imgs)-1))
sel = st.selectbox('Image', imgs, index=st.session_state['img_idx'] if imgs else None)
if imgs:
    # Sync index if user picks from dropdown
    if sel != imgs[st.session_state['img_idx']]:
        try:
            st.session_state['img_idx'] = imgs.index(sel)
        except ValueError:
            pass

# Navigation controls
col_nav1, col_nav2, col_nav3 = st.columns(3)
with col_nav1:
    if st.button('◀ Prev') and imgs:
        st.session_state['img_idx'] = max(0, st.session_state['img_idx'] - 1)
with col_nav2:
    if imgs:
        st.markdown(f"**{st.session_state['img_idx']+1} / {len(imgs)}**")
with col_nav3:
    if st.button('Next ▶') and imgs:
        st.session_state['img_idx'] = min(len(imgs)-1, st.session_state['img_idx'] + 1)

# Per-image top-class selection & hotkey mapping
top_classes = ['Permanent_Legal','Permanent_Illegal','Temporary_Legal','Temporary_Illegal','Natural_Area']
# Object classes for Phase 1 object-first training
object_classes = [
    'vehicle_car','vehicle_bike','vehicle_auto','vehicle_truck_bus','vehicle_tractor',
    'emergency_vehicle','pedestrian_walking','pedestrian_queue','beggar_squatter',
    'pandal_tent','stage_platform','idol_statue','flag_banner','religious_marker',
    'vendor_cart_thela','food_stall','kiosk_cabin','shop_house','barricade_fence',
    'portable_toilet','dustbin_dump','garbage_heap','water_tank_tap','bus_shelter_signage',
    'cctv_tower_drone_station','open_fire_stove','sand_heap','sewage_pipe_drain','boat',
    'hoarding_poster_banner'
]

mode = st.radio('Label mode', ['Objects', 'Top-Class (Legal categories)'],
                index=0 if st.session_state['label_mode'] == 'Objects' else 1, horizontal=True)
st.session_state['label_mode'] = mode
current_cls = st.session_state.get('image_class') or 'Temporary_Legal'
try:
    default_idx = top_classes.index(current_cls)
except ValueError:
    default_idx = 2
sel_cls = st.selectbox('Image-level class (optional)', top_classes, index=default_idx)
st.session_state['image_class'] = sel_cls

hk = st.text_input('Hotkey (1-5 class | a=prev | d=next)', '')
if hk:
    key = hk.strip().lower()[-1]
    if key in ['1','2','3','4','5']:
        st.session_state['image_class'] = top_classes[int(key)-1]
    elif key == 'a' and imgs:
        st.session_state['img_idx'] = max(0, st.session_state['img_idx'] - 1)
    elif key == 'd' and imgs:
        st.session_state['img_idx'] = min(len(imgs)-1, st.session_state['img_idx'] + 1)

# Drawing helper
def draw_boxes(img: Image.Image, boxes: List[Dict]):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    W, H = vis.size
    for b in boxes:
        cx, cy, w, h = b['bbox']
        x1 = (cx - w/2) * W
        y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W
        y2 = (cy + h/2) * H
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
    return vis

# Load image
if imgs and 0 <= st.session_state['img_idx'] < len(imgs):
    sel = imgs[st.session_state['img_idx']]
if sel and os.path.exists(sel):
    img = Image.open(sel).convert('RGB')
    st.session_state['current_image'] = sel
    W, H = img.size
    st.image(draw_boxes(img, st.session_state['boxes']), caption=sel, use_column_width=True)

    st.subheader('Add box (normalized)')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cx = st.number_input('cx', min_value=0.0, max_value=1.0, value=0.5)
    with c2:
        cy = st.number_input('cy', min_value=0.0, max_value=1.0, value=0.5)
    with c3:
        w = st.number_input('w', min_value=0.0, max_value=1.0, value=0.2)
    with c4:
        h = st.number_input('h', min_value=0.0, max_value=1.0, value=0.2)

    if mode == 'Objects':
        obj_idx = object_classes.index(st.session_state['object_class']) if st.session_state['object_class'] in object_classes else 5
        sel_obj = st.selectbox('Object class', object_classes, index=obj_idx)
        st.session_state['object_class'] = sel_obj
    structure_type = st.text_input('structure_type (free text)', 'other')
    if st.button('Add box'):
        attrs = {
            'permanence': permanence,
            'zone': zone,
            'permit_status': permit_status,
            'area_type': area_type,
            'structure_type': structure_type
        }
        if mode == 'Objects':
            attrs['object_class'] = st.session_state['object_class']
        else:
            attrs['top_class'] = st.session_state['image_class']
        st.session_state['boxes'].append({'bbox': [cx, cy, w, h], 'attributes': attrs})

    st.markdown('Or draw boxes with the mouse:')
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", stroke_width=2, stroke_color="#ff0000",
        background_image=img, height=min(720, H), width=min(1024, W), drawing_mode="rect",
        key="canvas",
    )
    # Convert drawn rectangles to normalized boxes when user clicks below
    if st.button('Add drawn boxes') and canvas_result and canvas_result.json_data:
        try:
            for obj in canvas_result.json_data["objects"]:
                if obj.get('type') != 'rect':
                    continue
                left = obj.get('left', 0)
                top = obj.get('top', 0)
                width = obj.get('width', 0)
                height = obj.get('height', 0)
                # Normalize
                nx = (left + width/2) / W
                ny = (top + height/2) / H
                nw = width / W
                nh = height / H
                attrs = {
                    'permanence': permanence,
                    'zone': zone,
                    'permit_status': permit_status,
                    'area_type': area_type,
                    'structure_type': structure_type
                }
                if mode == 'Objects':
                    attrs['object_class'] = st.session_state['object_class']
                else:
                    attrs['top_class'] = st.session_state['image_class']
                st.session_state['boxes'].append({'bbox': [nx, ny, nw, nh], 'attributes': attrs})
            st.success('Added drawn boxes')
        except Exception as e:
            st.error(f'Failed to add drawn boxes: {e}')

    if st.button('Undo last box') and st.session_state['boxes']:
        st.session_state['boxes'].pop()

    st.write('Current boxes:', st.session_state['boxes'])

    if st.button('Save to JSONL'):
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        rel = sel.split('dataset/')[-1] if 'dataset/' in sel else sel
        # Attach optional top_class for the whole image (applies to boxes unless overridden)
        for b in st.session_state['boxes']:
            b.setdefault('attributes', {})
            if mode == 'Objects':
                b['attributes'].setdefault('object_class', st.session_state['object_class'])
            else:
                b['attributes'].setdefault('top_class', st.session_state['image_class'])
        rec = {'image': rel, 'boxes': st.session_state['boxes']}
        with open(output_jsonl, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        st.success(f'Appended 1 record to {output_jsonl}')
        st.session_state['boxes'] = []
        # Auto-advance to next image
        if imgs:
            st.session_state['img_idx'] = min(len(imgs)-1, st.session_state['img_idx'] + 1)

    st.divider()
    st.subheader('Generate YOLO Labels and Train')
    cta1, cta2, cta3 = st.columns(3)
    with cta1:
        if st.button('Generate Top-Class YOLO Labels'):
            os.makedirs(labels_dir, exist_ok=True)
            cmd = [
                sys.executable,
                'tools/ontology_mapper.py',
                '--jsonl', output_jsonl,
                '--out', labels_dir,
                '--images-root', images_dir,
            ]
            try:
                subprocess.check_call(cmd)
                st.success(f'Top-class labels written to {labels_dir}')
            except subprocess.CalledProcessError as e:
                st.error(f'Label generation failed: {e}')
    with cta2:
        if st.button('Generate Object YOLO Labels'):
            os.makedirs(labels_dir, exist_ok=True)
            cmd = [
                sys.executable,
                'tools/jsonl_to_yolo_objects.py',
                '--jsonl', output_jsonl,
                '--out', labels_dir,
                '--data', objects_data_yaml,
            ]
            try:
                subprocess.check_call(cmd)
                st.success(f'Object labels written to {labels_dir}')
            except subprocess.CalledProcessError as e:
                st.error(f'Object label generation failed: {e}')
    with cta3:
        train_target = st.selectbox('Train target', ['Objects', 'Top-Class'], index=0)
        if st.button('Train YOLO (ultralytics)'):
            try:
                from ultralytics import YOLO
                model = YOLO('yolov8n.pt')
                chosen_yaml = objects_data_yaml if train_target == 'Objects' else data_yaml
                model.train(data=chosen_yaml, epochs=int(train_epochs), imgsz=int(imgsz), batch=16, patience=10)
                st.success('Training completed. Check runs/detect for results.')
            except Exception as e:
                st.error(f'Training failed: {e}')

    st.subheader('Export annotations to CSV')
    if st.button('Export CSV'):
        try:
            rows = []
            if os.path.exists(output_jsonl):
                with open(output_jsonl, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        img = rec.get('image')
                        for b in rec.get('boxes', []):
                            attrs = b.get('attributes', {})
                            cx, cy, w, h = b.get('bbox', [0,0,0,0])
                            rows.append({
                                'image': img,
                                'cx': cx, 'cy': cy, 'w': w, 'h': h,
                                'top_class': attrs.get('top_class'),
                                'permanence': attrs.get('permanence'),
                                'zone': attrs.get('zone'),
                                'permit_status': attrs.get('permit_status'),
                                'area_type': attrs.get('area_type'),
                                'structure_type': attrs.get('structure_type'),
                            })
            import pandas as pd
            df = pd.DataFrame(rows)
            os.makedirs(os.path.dirname(csv_export_path), exist_ok=True)
            df.to_csv(csv_export_path, index=False)
            st.success(f'Wrote {len(rows)} rows to {csv_export_path}')
        except Exception as e:
            st.error(f'CSV export failed: {e}')

else:
    st.info('No images found. Place images under the selected directory.')
