import numpy as np
import cv2
import h5py
from shapely.geometry import Polygon, Point
from svgpathtools import svg2paths2, parse_path
from lxml import etree
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import argparse

# --- Try CuPy, fallback to NumPy ---
try:
    import cupy as cp
    _ = cp.zeros(1)  # test GPU
    xp = cp
    GPU_AVAILABLE = True
    print("✅ Using GPU with CuPy")
except Exception:
    cp = np  # alias so code compiles
    xp = np
    GPU_AVAILABLE = False
    print("⚠️ CuPy/CUDA not available, falling back to CPU (NumPy)")

# --- Parameters ---
sigma = 25
max_value = 2000
canvas_width = 800
canvas_height = 1000
fps = 30
max_distance = 400
batch_size = 10000  # pixels per batch for memory efficiency

# --- SVG loading ---
def load_svg_points_and_contour(svg_file, is_right_hand=False):
    print(f"Loading SVG file: {svg_file}")
    paths, attributes, _ = svg2paths2(svg_file)
    outer_path_d = next((attr['d'] for attr in attributes if attr.get('id') == 'outerContour'), None)
    if outer_path_d is None:
        raise ValueError("SVG missing path with id='outerContour'")
    path_obj = parse_path(outer_path_d)
    outer_poly = Polygon([(seg.start.real, seg.start.imag) for seg in path_obj])
    tree = etree.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': root.nsmap[None]}
    source_points = {
        circle.get('id'): (float(circle.get('cx')), float(circle.get('cy')))
        for circle in root.findall('.//svg:circle', ns) if circle.get('id') is not None
    }
    source_points = dict(sorted(source_points.items()))
    if len(source_points) == 0:
        raise ValueError("No source points found in SVG")
    if is_right_hand:
        source_indices = [(15 - int(y), 15 - int(x)) for (x, y) in [sid.split('-') for sid in source_points]]
    else:
        source_indices = [(int(x), int(y)) for (x, y) in [sid.split('-') for sid in source_points]]
    return outer_poly, source_points, source_indices

def compute_transform(polygon, width, height, padding=10):
    minx, miny, maxx, maxy = polygon.bounds
    scale = min((width - 2 * padding) / (maxx - minx), (height - 2 * padding) / (maxy - miny))
    tx, ty = -minx * scale + padding, -miny * scale + padding
    return scale, tx, ty

def apply_transform(polygon, points_dict, scale, tx, ty):
    poly = Polygon([(x * scale + tx, y * scale + ty) for x, y in polygon.exterior.coords])
    points = {k: (x * scale + tx, y * scale + ty) for k, (x, y) in points_dict.items()}
    return poly, points

def get_pixels_in_contour(polygon, width, height):
    xs, ys = np.arange(width), np.arange(height)
    xv, yv = np.meshgrid(xs, ys)
    points = [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
    mask = np.array([polygon.contains(pt) for pt in points])
    return np.stack([xv.ravel()[mask], yv.ravel()[mask]], axis=1)

# --- Chunked weights computation (GPU or CPU) ---
def compute_weights_visibility_aware_chunked(pixel_coords, source_points, polygon, sigma, max_dist, batch_size=10000):
    print(f"Starting visibility-aware weight computation ({'GPU' if GPU_AVAILABLE else 'CPU'})...")
    n_pixels = pixel_coords.shape[0]
    n_sources = len(source_points)
    weights = xp.zeros((n_pixels, n_sources), dtype=xp.float32)

    px_all = xp.asarray(pixel_coords[:,0], dtype=xp.float32)
    py_all = xp.asarray(pixel_coords[:,1], dtype=xp.float32)
    source_arr = xp.asarray(list(source_points.values()), dtype=xp.float32)

    coords = xp.asarray(np.array(polygon.exterior.coords), dtype=xp.float32)
    edges_start = coords[:-1]; edges_end = coords[1:]

    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay)*(bx-ax) > (by - ay)*(cx-ax)

    for start in tqdm(range(0, n_pixels, batch_size)):
        end = min(start + batch_size, n_pixels)
        px = px_all[start:end, None]; py = py_all[start:end, None]

        dx = px - source_arr[None,:,0]; dy = py - source_arr[None,:,1]
        dist = xp.sqrt(dx**2 + dy**2)
        mask = dist <= max_dist if max_dist>0 else xp.ones_like(dist, dtype=bool)

        vis = xp.ones_like(mask)
        for ex1, ey1, ex2, ey2 in zip(edges_start[:,0], edges_start[:,1], edges_end[:,0], edges_end[:,1]):
            sx = source_arr[:,0][None,:]; sy = source_arr[:,1][None,:]
            intersect = (
                ccw(px, py, ex1, ey1, ex2, ey2) != ccw(sx, sy, ex1, ey1, ex2, ey2)
            ) & (
                ccw(px, py, sx, sy, ex1, ey1) != ccw(px, py, sx, sy, ex2, ey2)
            )
            vis &= ~intersect

        final_mask = mask & vis
        w = xp.exp(-(dist**2)/(2*sigma**2)) * final_mask
        sum_w = xp.sum(w, axis=1, keepdims=True)
        valid_sum_mask = sum_w > 1e-6
        w_normalized = xp.zeros_like(w)
        w_normalized[valid_sum_mask.flatten()] = w[valid_sum_mask.flatten()] / sum_w[valid_sum_mask.flatten()]
        weights[start:end] = w_normalized

    return weights

def values_to_bgr(values_arr):
    values = xp.asnumpy(values_arr) if GPU_AVAILABLE else values_arr
    values = values.astype(np.float32)
    bgr = np.zeros((values.shape[0], 3), dtype=np.uint8)

    no_data_mask = values < 0
    no_data_color = np.array([192, 192, 192], dtype=np.uint8)
    bgr[no_data_mask] = no_data_color

    valid_data_mask = ~no_data_mask
    valid_values = values[valid_data_mask]

    v_norm = valid_values / max_value
    v_clipped = np.clip(v_norm, 0.0, 1.0)

    hsv_valid = np.zeros((valid_values.shape[0], 1, 3), dtype=np.float32)
    hsv_valid[..., 0] = (v_clipped * 240.0).reshape(-1, 1)
    hsv_valid[..., 1] = 1.0
    hsv_valid[..., 2] = 1.0

    valid_bgr = cv2.cvtColor(hsv_valid, cv2.COLOR_HSV2BGR).reshape(-1, 3) * 255
    bgr[valid_data_mask] = valid_bgr.astype(np.uint8)

    full_v_norm = np.zeros_like(values)
    full_v_norm[valid_data_mask] = v_norm
    bgr[full_v_norm > 1.0] = no_data_color

    return bgr.astype(np.uint8)

def render_frame(pressure_frame, weights, pixel_coords, width, height, source_indices):
    pressure_flat = np.array([pressure_frame[row, col] for (row, col) in source_indices], dtype=np.float32)
    pixel_vals = weights @ (xp.asarray(pressure_flat) if GPU_AVAILABLE else pressure_flat)
    sum_of_weights = xp.sum(weights, axis=1)
    no_signal_mask = sum_of_weights < 1e-6
    pixel_vals[no_signal_mask] = -1
    colors_bgr = values_to_bgr(pixel_vals)
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    px = np.clip(pixel_coords[:, 0], 0, width - 1).astype(int)
    py = np.clip(pixel_coords[:, 1], 0, height - 1).astype(int)
    img[py, px] = colors_bgr
    return img

def make_frame_processor(weights, pixel_coords, width, height, source_indices, ts_array, pressure_array):
    def interpolate_pressure(ts_array, pressure_array, target_ts):
        idx = np.searchsorted(ts_array, target_ts)
        if idx == 0: return pressure_array[0]
        if idx >= len(ts_array): return pressure_array[-1]
        t0, t1 = ts_array[idx-1], ts_array[idx]
        p0, p1 = pressure_array[idx-1], pressure_array[idx]
        alpha = (target_ts - t0) / (t1 - t0)
        return (1-alpha)*p0 + alpha*p1

    def process_single_frame(t):
        pressure_frame = interpolate_pressure(ts_array, pressure_array, t)
        return render_frame(pressure_frame, weights, pixel_coords, width, height, source_indices)

    return process_single_frame

def load_glove_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        frame_count = f['frame_count'][0]
        ts = np.array(f['ts'][:frame_count])
        pressure = np.array(f['pressure'][:frame_count])
    return ts, pressure

def get_cache_path(svg_file, is_right_hand, sigma, max_distance, width, height):
    basename = os.path.basename(svg_file).split('.')[0]
    side = 'right' if is_right_hand else 'left'
    return f'weights_cache_{basename}_{side}_s{sigma}_d{max_distance}_w{width}_h{height}.npz'

def prepare_renderer(h5_path, svg_file, is_right_hand):
    print(f"Preparing renderer for {'right' if is_right_hand else 'left'} hand with HDF5: {h5_path} and SVG: {svg_file}")
    
    cache_path = get_cache_path(svg_file, is_right_hand, sigma, max_distance, canvas_width, canvas_height)
    outer_poly, source_points, source_indices = load_svg_points_and_contour(svg_file, is_right_hand=is_right_hand)
    scale, tx, ty = compute_transform(outer_poly, canvas_width, canvas_height)
    outer_poly, source_points = apply_transform(outer_poly, source_points, scale, tx, ty)

    if os.path.exists(cache_path):
        print(f"Loading precomputed weights from: {cache_path}")
        cache = np.load(cache_path)
        weights = xp.asarray(cache['weights']) if GPU_AVAILABLE else cache['weights']
        pixel_coords = cache['pixel_coords']
    else:
        pixel_coords = get_pixels_in_contour(outer_poly, canvas_width, canvas_height)
        weights = compute_weights_visibility_aware_chunked(
            pixel_coords, source_points, outer_poly, sigma, max_distance, batch_size=batch_size)
        np.savez_compressed(cache_path, weights=(cp.asnumpy(weights) if GPU_AVAILABLE else weights), pixel_coords=pixel_coords)
        print(f"Saved weights cache to: {cache_path}")

    ts_array, pressure_array = load_glove_data(h5_path)
    return make_frame_processor(weights, pixel_coords, canvas_width, canvas_height,
                                source_indices, ts_array, pressure_array), ts_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_h5', type=str, help='Path to left glove HDF5 file')
    parser.add_argument('--right_h5', type=str, help='Path to right glove HDF5 file')
    parser.add_argument('--small', action='store_true', help='Use small glove SVG if set')
    parser.add_argument('--output', type=str, default='tactile_overlay.mp4')
    args = parser.parse_args()

    assert args.left_h5 or args.right_h5, "At least one of --left_h5 or --right_h5 must be provided"

    small_svg = "source_points_left_small.svg"
    large_svg = "source_points_left_large.svg"

    if args.left_h5:
        left_svg = small_svg if args.small else large_svg
        left_renderer, ts_left = prepare_renderer(args.left_h5, left_svg, is_right_hand=False)
    if args.right_h5:
        right_svg = small_svg if args.small else large_svg
        right_renderer, ts_right = prepare_renderer(args.right_h5, right_svg, is_right_hand=True)

    if args.left_h5 and args.right_h5:
        ts_min = max(ts_left[0], ts_right[0])
        ts_max = min(ts_left[-1], ts_right[-1])
        uniform_ts = np.arange(ts_min, ts_max, 1 / fps)
    elif args.left_h5:
        uniform_ts = np.arange(ts_left[0], ts_left[-1], 1 / fps)
    else:
        uniform_ts = np.arange(ts_right[0], ts_right[-1], 1 / fps)

    out_width = canvas_width * (2 if (args.left_h5 and args.right_h5) else 1)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, canvas_height))

    with ThreadPoolExecutor(max_workers=4) as executor:
        with tqdm(total=len(uniform_ts), desc="Rendering frames") as pbar:
            for t in uniform_ts:
                if args.left_h5 and args.right_h5:
                    left_frame = left_renderer(t)
                    right_frame = right_renderer(t)
                    combined = np.hstack([left_frame, right_frame])
                elif args.left_h5:
                    combined = left_renderer(t)
                else:
                    combined = right_renderer(t)
                writer.write(combined)
                pbar.update()

    writer.release()
    print(f"Saved video to {args.output}")

if __name__ == "__main__":
    main()
