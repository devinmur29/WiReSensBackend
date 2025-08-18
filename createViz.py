import h5py
import json
import numpy as np
import cairo
import imageio.v2 as imageio
from xml.etree import ElementTree as ET

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

def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        rp, gp, bp = c, x, 0
    elif h < 120:
        rp, gp, bp = x, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, x
    elif h < 240:
        rp, gp, bp = 0, x, c
    elif h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x

    r, g, b = rp + m, gp + m, bp + m
    return (r, g, b)

def value_to_color(val):
    clamped = np.clip(val, MIN_PRESSURE, MAX_PRESSURE)
    norm = (clamped - MIN_PRESSURE) / (MAX_PRESSURE - MIN_PRESSURE)
    hue = (1 - norm) * 240  # 240 = blue, 0 = red
    return hsv_to_rgb(hue, 1.0, 1.0)

def create_video(left_h5=None, right_h5=None, mapping_json=None, svg_file=None,
                 output_mp4="output.mp4", use_normalized=True,
                 width=800, height=800):

    assert left_h5 or right_h5, "At least one of left_h5 or right_h5 must be provided."

    mapping = json.load(open(mapping_json))
    voronoi_polygons = parse_svg(svg_file)
    polygon_labels = list(voronoi_polygons.keys())

    # Extract hand outline polygon points for clipping
    hand_outline_points = voronoi_polygons.get("hand_outline", None)
    if hand_outline_points is None:
        print("Warning: No 'hand_outline' polygon found in SVG. No clipping will be applied.")

    left_data = left_ts = right_data = right_ts = None
    all_left_interp = []
    all_right_interp = []

    if left_h5:
        left_data, left_ts = load_data(left_h5)
        left_weights = precompute_weights(mapping, is_left=True)
        left_min = np.min(left_data, axis=0)
        left_max = np.max(left_data, axis=0)
        left_range = left_max - left_min
        left_range_safe = np.where(left_range > 1e-5, left_range, 1)  # Avoid div zero

    if right_h5:
        right_data, right_ts = load_data(right_h5)
        right_weights = precompute_weights(mapping, is_left=False)
        right_min = np.min(right_data, axis=0)
        right_max = np.max(right_data, axis=0)
        right_range = right_max - right_min
        right_range_safe = np.where(right_range > 1e-5, right_range, 1)

    # Time alignment
    if left_ts is not None and right_ts is not None:
        start = max(left_ts[0], right_ts[0])
        end = min(left_ts[-1], right_ts[-1])
    elif left_ts is not None:
        start = left_ts[0]
        end = left_ts[-1]
    elif right_ts is not None:
        start = right_ts[0]
        end = right_ts[-1]
    else:
        raise RuntimeError("No timestamps available")

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

    print(f"Rendering and encoding {frame_count} frames with pycairo + imageio...")

    # Prepare polygon coordinate arrays for Cairo
    left_polys = [list(reversed(v)) for v in voronoi_polygons.values()]
    right_polys = [v for v in voronoi_polygons.values()]

    def draw_polys(ctx, polys, interp, use_norm):
        for pid, coords in zip(polygon_labels, polys):
            v = interp.get(pid, MIN_PRESSURE)
            if use_norm and v >= NORMALIZED_VIS_THRESHOLD:
                color = FALLBACK_COLOR[:3]
            else:
                color = value_to_color(v)
            ctx.set_source_rgb(*color)
            ctx.move_to(*coords[0])
            for x, y in coords[1:]:
                ctx.line_to(x, y)
            ctx.close_path()
            ctx.fill()

    frames = []
    for i in range(frame_count):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, width, height)
        ctx = cairo.Context(surface)
        # Fill background white
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Apply clipping to hand outline if available
        if hand_outline_points:
            ctx.new_path()
            ctx.move_to(*hand_outline_points[0])
            for pt in hand_outline_points[1:]:
                ctx.line_to(*pt)
            ctx.close_path()
            ctx.clip()

        if left_data is not None:
            draw_polys(ctx, left_polys, all_left_interp[i], use_normalized)
        if right_data is not None:
            draw_polys(ctx, right_polys, all_right_interp[i], use_normalized)

        buf = surface.get_data()
        frame = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
        frame_rgb = frame[:, :, :3]
        frames.append(frame_rgb.copy())

        if i % 50 == 0:
            print(f"Rendered frame {i}/{frame_count}")

    imageio.mimsave(output_mp4, frames, fps=FPS, codec="libx264", quality=8)
    print(f"Video saved to {output_mp4}")

if __name__ == "__main__":
    create_video(
        right_h5="./recordings/smallglovedemo.hdf5",
        mapping_json="point_weight_mappings_small.json",
        svg_file="voronoi_regions_small.svg",
        output_mp4="pressure_large_small_clipped.mp4",
        use_normalized=False,
        width=800,
        height=800
    )
