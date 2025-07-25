import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib
from xml.etree import ElementTree as ET

matplotlib.use("Agg")  # no GUI backend

FPS = 20
MAX_PRESSURE, MIN_PRESSURE = 3000, 0
NORMALIZED_VIS_THRESHOLD = 2500
FALLBACK_COLOR = (0.2, 0.2, 0.2, 1.0)  # dark gray RGBA


def load_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        fc = f['frame_count'][0]
        ts = np.array(f['ts'][:fc])
        pressure = np.array(f['pressure'][:fc]).astype(np.float32)
    return pressure, ts

def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    polygons = {}
    for poly in root.findall('.//svg:polygon', ns):
        pid = poly.attrib['id']
        points = [tuple(map(float, p.split(','))) for p in poly.attrib['points'].strip().split()]
        polygons[pid] = points
    return polygons

def precompute_weights(mapping, is_left=True):
    weights = {}
    for label, neighbors in mapping.items():
        matrix = np.zeros((16, 16), dtype=np.float32)
        total_weight = 0.0
        for q in ['NE', 'NW', 'SW', 'SE']:
            source_id, dist = neighbors[q]
            if source_id != 'N/A':
                y, x = map(int, source_id.split('-'))
                if not is_left:
                    y, x = 15 - x, 15 - y
                weight = 1 / dist if dist and dist > 0.001 else 1e6
                matrix[y, x] += weight
                total_weight += weight
        if total_weight > 0:
            matrix /= total_weight
        weights[label] = matrix
    return weights

def interpolate_fast(pressure16x16, precomputed_weights):
    out = {}
    for label, W in precomputed_weights.items():
        out[label] = np.sum(W * pressure16x16)
    return out

def value_to_color(val):
    clamped = np.clip(val, MIN_PRESSURE, MAX_PRESSURE)
    norm = (clamped - MIN_PRESSURE) / (MAX_PRESSURE - MIN_PRESSURE)
    hue = (1 - norm) * 240
    # hsv colormap returns RGBA float tuple, compatible with PolyCollection
    return plt.cm.hsv(hue / 360)




def create_video(left_h5=None, right_h5=None, mapping_json=None, svg_file=None, output_mp4="output.mp4", use_normalized=True):
    from matplotlib.animation import FFMpegWriter

    assert left_h5 or right_h5, "At least one of left_h5 or right_h5 must be provided."
    
    mapping = json.load(open(mapping_json))
    voronoi_polygons = parse_svg(svg_file)
    polygon_labels = list(voronoi_polygons.keys())

    left_data = left_ts = right_data = right_ts = None
    all_left_interp = []
    all_right_interp = []

    if left_h5:
        left_data, left_ts = load_data(left_h5)
        left_weights = precompute_weights(mapping, is_left=True)
        left_min = np.min(left_data, axis=0)
        left_max = np.max(left_data, axis=0)
        left_range = left_max - left_min
        left_range_safe = np.where(left_range > 1e-5, left_range, 1)  # Avoid divide-by-zero

    if right_h5:
        right_data, right_ts = load_data(right_h5)
        right_weights = precompute_weights(mapping, is_left=False)
        right_min = np.min(right_data, axis=0)
        right_max = np.max(right_data, axis=0)
        right_range = right_max - right_min
        right_range_safe = np.where(right_range > 1e-5, right_range, 1)


    # Time alignment
    timestamps = None
    if left_ts is not None and right_ts is not None:
        start = max(left_ts[0], right_ts[0])
        end = min(left_ts[-1], right_ts[-1])
    elif left_ts is not None:
        start = left_ts[0]
        end = left_ts[-1]
    elif right_ts is not None:
        start = right_ts[0]
        end = right_ts[-1]

    frame_count = int((end - start) * FPS)
    timestamps = np.linspace(start, end, frame_count)

    print(f"Precomputing {frame_count} {'normalized' if use_normalized else 'raw'} frames...")

    for t in timestamps:
        if left_data is not None:
            lf_idx = np.searchsorted(left_ts, t)
            left_raw = left_data[min(lf_idx, len(left_data) - 1)]
            if use_normalized:
                left_norm = (left_raw - left_min) / left_range_safe
                left_scaled = left_norm * (MAX_PRESSURE - MIN_PRESSURE) + MIN_PRESSURE

            else:
                left_scaled = np.copy(left_raw)
            all_left_interp.append(interpolate_fast(left_scaled, left_weights))

        if right_data is not None:
            rt_idx = np.searchsorted(right_ts, t)
            right_raw = right_data[min(rt_idx, len(right_data) - 1)]
            if use_normalized:
                right_norm = (right_raw - right_min) / right_range_safe
                right_scaled = right_norm * (MAX_PRESSURE - MIN_PRESSURE) + MIN_PRESSURE

            else:
                right_scaled = np.copy(right_raw)
            all_right_interp.append(interpolate_fast(right_scaled, right_weights))

    # Set up figure
    has_left = left_data is not None
    has_right = right_data is not None
    ncols = int(has_left) + int(has_right)
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 8), dpi=80)
    if ncols == 1:
        axs = [axs]

    idx = 0
    if has_left:
        left_polys = [list(reversed(v)) for v in voronoi_polygons.values()]
        left_collection = PolyCollection(left_polys, edgecolors='none')
        axs[idx].add_collection(left_collection)
        xs = [x for poly in left_polys for (x, y) in poly]
        ys = [y for poly in left_polys for (x, y) in poly]
        axs[idx].set_xlim(min(xs) - 10, max(xs) + 10)
        axs[idx].set_ylim(max(ys) + 10, min(ys) - 10)
        axs[idx].set_aspect('equal')
        axs[idx].axis('off')
        idx += 1
    if has_right:
        right_polys = [v for v in voronoi_polygons.values()]
        right_collection = PolyCollection(right_polys, edgecolors='none')
        xs = [x for poly in right_polys for (x, y) in poly]
        ys = [y for poly in right_polys for (x, y) in poly]
        axs[idx].set_xlim(min(xs) - 10, max(xs) + 10)
        axs[idx].set_ylim(max(ys) + 10, min(ys) - 10)
        axs[idx].add_collection(right_collection)
        axs[idx].set_aspect('equal')
        axs[idx].axis('off')

    # Write video
    metadata = dict(title='Glove Pressure Visualization', artist='Your Name')
    writer = FFMpegWriter(fps=FPS, metadata=metadata, codec='libx264', extra_args=['-preset', 'ultrafast'])

    print(f"Writing video to {output_mp4}...")
    with writer.saving(fig, output_mp4, dpi=80):
        for i in range(frame_count):
            if has_left:
                left_colors = []
                for k in polygon_labels:
                    v = all_left_interp[i].get(k, MIN_PRESSURE)
                    if use_normalized and v >= NORMALIZED_VIS_THRESHOLD:
                        left_colors.append(FALLBACK_COLOR)
                    else:
                        left_colors.append(value_to_color(v))
                left_collection.set_facecolors(left_colors)

            if has_right:
                right_colors = []
                for k in polygon_labels:
                    v = all_right_interp[i].get(k, MIN_PRESSURE)
                    if use_normalized and v >= NORMALIZED_VIS_THRESHOLD:
                        right_colors.append(FALLBACK_COLOR)
                    else:
                        right_colors.append(value_to_color(v))
                right_collection.set_facecolors(right_colors)

            writer.grab_frame()
            if i % 50 == 0:
                print(f"Frame {i+1}/{frame_count}")

    print("Video rendering complete.")


if __name__ == "__main__":
    create_video(
        left_h5="./recordings/recentLeft.hdf5",
        mapping_json="point_weight_mappings_large.json",
        svg_file="voronoi_regions_large.svg",
        output_mp4="glove_viz_pose.mp4",
        use_normalized=False
    )
