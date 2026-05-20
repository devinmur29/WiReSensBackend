import numpy as np
import cv2
import h5py
from shapely.geometry import Polygon
from svgpathtools import svg2paths2, parse_path
from lxml import etree
from tqdm import tqdm

# Try CuPy, fallback to NumPy
try:
    import cupy as cp
    _ = cp.zeros(1)
    xp = cp
    GPU_AVAILABLE = True
    print("✅ Using GPU with CuPy")
except Exception:
    cp = np
    xp = np
    GPU_AVAILABLE = False
    print("⚠️ CuPy/CUDA not available, falling back to CPU (NumPy)")

sigma_x = 18.0
sigma_y = 18.0
max_value = 2800
canvas_width = 800
canvas_height = 1000
fps = 30


def load_svg_points_and_contour(svg_file, is_right_hand=False):
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

def load_glove_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        frame_count = f['frame_count'][0]
        ts = np.array(f['ts'][:frame_count])
        pressure = np.array(f['pressure'][:frame_count])
    return ts, pressure

def precompute_gaussian_kernels(source_points):
    """Precompute Gaussian kernels for each source point"""
    y_coords, x_coords = xp.meshgrid(xp.arange(canvas_height), xp.arange(canvas_width), indexing="ij")
    kernels = {}
    for point_id, (x0, y0) in source_points.items():
        dx = x_coords - x0
        dy = y_coords - y0
        kernel = xp.exp(-((dx**2)/(2*sigma_x**2) + (dy**2)/(2*sigma_y**2)))
        kernels[point_id] = kernel
    return kernels

def render_gaussian_video(h5_path, svg_file, output_file, is_right_hand=False):
    outer_poly, source_points, source_indices = load_svg_points_and_contour(svg_file, is_right_hand)
    scale, tx, ty = compute_transform(outer_poly, canvas_width, canvas_height)
    outer_poly, source_points = apply_transform(outer_poly, source_points, scale, tx, ty)
    ts, pressure = load_glove_data(h5_path)  # shape: (frames, rows, cols)
    n_frames = pressure.shape[0]

    print("Precomputing Gaussian kernels...")
    kernels = precompute_gaussian_kernels(source_points)

    # Precompute polygon mask
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    polygon_pts = np.array(list(outer_poly.exterior.coords)).round().astype(np.int32)
    cv2.fillPoly(mask, [polygon_pts], color=1)  # mask: 1 inside polygon, 0 outside

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (canvas_width, canvas_height))

    print(f"Rendering {n_frames} frames to {output_file}...")
    for frame_idx in tqdm(range(n_frames), desc="Frames"):
        frame_img = xp.zeros((canvas_height, canvas_width), dtype=xp.float32)

        for idx, point_id in enumerate(source_points.keys()):
            row, col = source_indices[idx]
            press_val = pressure[frame_idx, row, col]
            if press_val > max_value:
                press_val = max_value
            amplitude = max_value - press_val
            frame_img += amplitude * kernels[point_id]

        frame_img = xp.asnumpy(frame_img)

        inside_mask = mask.astype(bool)
        outside_mask = ~inside_mask

        adaptive_epsilon = max_value*4*0.1

        clipped = np.clip(frame_img, adaptive_epsilon, max_value)
        gamma = 0.6  # less than 1 spreads low values
        normalized = (((clipped-adaptive_epsilon) / (max_value-adaptive_epsilon)) ** gamma * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        final_frame = np.ones_like(heatmap, dtype=np.uint8) * 255
        final_frame[outside_mask] = [255, 255, 255]

        # Adaptive epsilon threshold per frame
        
        zero_mask = (frame_img < adaptive_epsilon) & inside_mask
        heatmap[zero_mask] = [128, 128, 128]  # gray

        final_frame[inside_mask] = heatmap[inside_mask]
        video_writer.write(final_frame)

    video_writer.release()
    print(f"✅ RGB heatmap clipped to polygon saved to {output_file}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--svg_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--right_hand", action="store_true")
    args = parser.parse_args()

    render_gaussian_video(args.h5_path, args.svg_file, args.output_file, args.right_hand)
