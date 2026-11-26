"""Chạy BÊN TRONG Blender: thực thi một script bpy, đo thống kê mesh, lấy mẫu điểm, (tùy chọn) render 4 góc.

blender -b --factory-startup --python blender_exec.py -- <code.py> <out_dir> [--render] [--res 256]
        [--samples 8] [--points 4096] [--threads 2]

Ghi <out_dir>/result.json; points.npy (điểm bề mặt đã chuẩn hóa về hộp [-1,1]^3);
view_045/135/225/315.png (RGBA, nền trong suốt, material xám đồng nhất) nếu --render.
Chuẩn render: azimuth 45/135/225/315°, elevation 25°, FOV 40°, khung hình bao trọn vật thể.
"""

import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CODE_PATH, OUT_DIR = argv[0], argv[1]
OPTS = argv[2:]


def opt(name, default):
    if name in OPTS:
        return type(default)(OPTS[OPTS.index(name) + 1])
    return default


RENDER = "--render" in OPTS
RES = opt("--res", 256)
SAMPLES = opt("--samples", 8)
NPTS = opt("--points", 4096)
THREADS = opt("--threads", 2)
GEOM_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD"}

os.makedirs(OUT_DIR, exist_ok=True)
res = {"ok": False, "error": None, "n_mesh_objects": 0, "tris": 0, "exec_s": None, "render_s": None,
       "blender": bpy.app.version_string}


def finish():
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(res, f)
    sys.stdout.flush()
    os._exit(0)


# ---------------------------------------------------------------- 1. thực thi
bpy.ops.wm.read_factory_settings(use_empty=True)
src = open(CODE_PATH, encoding="utf-8", errors="replace").read()
t0 = time.time()
try:
    exec(compile(src, CODE_PATH, "exec"), {"__name__": "__main__", "__file__": CODE_PATH})
except BaseException as e:  # gồm cả SystemExit do script tự gọi
    if not (isinstance(e, SystemExit) and e.code in (None, 0)):
        res["error"] = f"{type(e).__name__}: {e}"[:2000]
        res["traceback"] = traceback.format_exc()[-4000:]
res["exec_s"] = round(time.time() - t0, 2)

# ---------------------------------------------------------------- 2. thống kê hình học
try:
    dg = bpy.context.evaluated_depsgraph_get()
    chunks, n_obj, total = [], 0, 0
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type not in GEOM_TYPES:
            continue
        try:
            me = ob.to_mesh()
        except Exception:
            continue
        if me is None:
            continue
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nt:
            n_obj += 1
            total += nt
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            tri = np.empty(nt * 3, dtype=np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            M = np.array(inst.matrix_world, dtype=np.float32)
            wco = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            chunks.append(wco[tri.reshape(-1, 3)])
        ob.to_mesh_clear()
    res["n_mesh_objects"], res["tris"] = n_obj, int(total)

    if chunks:
        T = np.concatenate(chunks)
        flat = T.reshape(-1, 3)
        mn, mx = flat.min(0), flat.max(0)
        center = (mn + mx) / 2
        extent = float((mx - mn).max())
        scale = 2.0 / max(extent, 1e-6)
        res["bbox_size"] = [float(x) for x in (mx - mn)]
        Tn = (T - center) * scale
        a, b = Tn[:, 1] - Tn[:, 0], Tn[:, 2] - Tn[:, 0]
        area = np.linalg.norm(np.cross(a, b), axis=1)
        if area.sum() > 0:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(Tn), size=NPTS, p=area / area.sum())
            u, v = rng.random((NPTS, 1)), rng.random((NPTS, 1))
            flip = (u + v) > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            P = Tn[idx, 0] + u * a[idx] + v * b[idx]
            np.save(os.path.join(OUT_DIR, "points.npy"), P.astype(np.float32))
        del T, Tn, chunks
except BaseException as e:
    res["error"] = res["error"] or f"STATS {type(e).__name__}: {e}"[:2000]

res["ok"] = res["error"] is None and res["tris"] > 0
if not (RENDER and res["ok"]):
    finish()

# ---------------------------------------------------------------- 3. render 4 góc
try:
    t1 = time.time()
    sc = bpy.context.scene
    for o in list(sc.objects):
        if o.type in {"LIGHT", "CAMERA"}:
            bpy.data.objects.remove(o, do_unlink=True)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = False
    sc.render.threads_mode = "FIXED"
    sc.render.threads = THREADS
    sc.render.resolution_x = sc.render.resolution_y = RES
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"

    world = bpy.data.worlds.new("t2a_world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = 0.6

    mat = bpy.data.materials.new("t2a_grey")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1)
    sc.view_layers[0].material_override = mat

    sun = bpy.data.objects.new("t2a_sun", bpy.data.lights.new("t2a_sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sc.collection.objects.link(sun)

    cam_data = bpy.data.cameras.new("t2a_cam")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(40)
    cam = bpy.data.objects.new("t2a_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    c = Vector([float(x) for x in center])
    d = 5.2 * (extent / 2)  # bán kính hình cầu bao (√3·half) / sin(20°) ≈ 5.06·half
    cam_data.clip_start, cam_data.clip_end = max(d * 0.01, 1e-4), d * 10
    el = math.radians(25)
    for az in (45, 135, 225, 315):
        a_ = math.radians(az)
        pos = c + Vector((d * math.cos(el) * math.cos(a_), d * math.cos(el) * math.sin(a_), d * math.sin(el)))
        cam.location = pos
        cam.rotation_euler = (c - pos).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = os.path.join(OUT_DIR, f"view_{az:03d}.png")
        bpy.ops.render.render(write_still=True)
    res["render_s"] = round(time.time() - t1, 2)
except BaseException as e:
    res["render_error"] = f"{type(e).__name__}: {e}"[:2000]

finish()
