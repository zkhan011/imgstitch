import cv2
import numpy as np
import os
import time
import threading
from collections import deque

# =========================
# CONFIG
# =========================
RTSP_HOST = "10.11.89.130"
RTSP_USER = "admin"
RTSP_PASS = "admin"
RTSP_PROFILE = "profile2/media.smp"
INPUT = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}/{RTSP_PROFILE}"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ROI = None   # e.g. (80, 220, 1900, 980)

SLIT_X = -1
SLIT_Y = -1

TARGET_FPS = 30

# =========================
# STITCH MODE
# =========================
AXIS = "x"       # x = side profile stitching
DIR = +1         # +1 append right, -1 append left

# =========================
# STITCHING / OVERLAP CONTROL
# =========================
USE_FEATURE_MATCHING = False
USE_FIXED_STRIP_ONLY = True   # matches your working approach: constant strip every captured frame
USE_EVERY_FRAME_STITCH = True
USE_SMART_NON_OVERLAP = True  # accumulate motion and append only new, non-overlapping content
AUTO_DIRECTION_FROM_DX = False  # keep UI-selected direction stable unless explicitly enabled

# seed strip used only for the first captured frame
FIXED_STRIP_W = 100
FIXED_STRIP_H = 100

# old seam overlap is not needed when using feature-matched strip width
SEAM_OVERLAP_PX = 0
STRIP_EMA_ALPHA = 0.0

# feature matching params
ORB_NFEATURES = 1800
ORB_SCALE_FACTOR = 1.2
ORB_NLEVELS = 8
MATCH_RATIO_TEST = 0.72
MIN_GOOD_MATCHES = 20
MAX_MATCH_STEP_X = 140
MAX_MATCH_STEP_Y = 140
MIN_FEATURE_STEP = 2
FEATURE_BAND_FRAC = 0.35     # lower band is typically more stable on conveyor/profile scans
FEATURE_IGNORE_TOP_PX = 0
MAD_OUTLIER_THR = 12.0       # tighter outlier rejection
FALLBACK_TO_FIXED_IF_NO_MATCH = True

# Optional smoothing of matched step
STEP_EMA_ALPHA = 0.15
STEP_SCALE = 1.00            # 1.0 keeps real-world length; <1.0 compresses/shrinks profile
MIN_EFFECTIVE_STEP = 2       # keep small but non-zero strips so every captured frame can be stitched
MAX_EFFECTIVE_STEP = 90      # cap strip growth to avoid occasional bad jumps
USE_PHASE_CORR = True        # robust fallback on repetitive container textures
MIN_PHASE_RESPONSE = 0.10
USE_STITCH_BAND = False      # disabled by default to match your baseline full-ROI stitching
STITCH_TOP_FRAC = 0.22
STITCH_BOTTOM_FRAC = 0.95

# Baseline fixed-strip mode (from your working version)
STRIP_W = 95
STRIP_H = 95
SEAM_OVERLAP_PX = 6
MIN_APPEND_STEP = 3
MAX_APPEND_STEP = 120
MIN_CAPTURE_SEC_BEFORE_STOP_CHECK = 6.0
DEBUG_LOG_EVERY_N_FRAMES = 8

# =========================
# OUTPUT
# =========================
SAVE_SCALE = 1.0

FINAL_PAD_TOP = 10
FINAL_PAD_BOTTOM = 10
FINAL_PAD_LEFT = 10
FINAL_PAD_RIGHT = 10

MIN_SAVE_W = 260
MIN_SAVE_H = 220

# =========================
# MOTION DETECTION
# =========================
MOTION_DS_W = 320
MOTION_BLUR = 5
MOTION_DIFF_THR = 14
MOTION_AREA_FRAC = 0.012

# start immediately
MOTION_START_SEC = 0.0

# stop after 2 sec no motion
NO_MOTION_STOP_SEC = 2.0

MOTION_USE_BOTTOM_FRAC = 0.45
IGNORE_TOP_PX = 0
AUTO_PREROLL_FRAMES = 8      # include a few frames before motion trigger
AUTO_POSTROLL_FRAMES = 8     # keep a few frames after motion disappears

# =========================
# FRAME BUFFER / STREAM ROBUSTNESS
# =========================
FRAMEBUFFER_MAX = 0
CAP_BUFFERSIZE = 64

READ_FAIL_RETRY_SEC = 0.20
REOPEN_STREAM_ON_FAIL = True
MAX_CONSECUTIVE_READ_FAILS = 20

# =========================
# GPU / CPU
# =========================
USE_GPU = True
GPU_DEVICE_ID = 0

# =========================
# UI
# =========================
WINDOW = "SideProfileScanner"
FONT = cv2.FONT_HERSHEY_SIMPLEX
SHOW_DEBUG = True

BTN_START = (10, 10, 140, 55)
BTN_STOP  = (155, 10, 285, 55)
BTN_SAVE  = (300, 10, 430, 55)
BTN_RESET = (445, 10, 575, 55)

# =========================
# FIRST RUN CONFIG
# =========================
CONFIG_WIN = "Config (first run) - ROI / Direction / Append"
CONF_AXIS_X = (20, 20, 240, 70)
CONF_AXIS_Y = (260, 20, 480, 70)
CONF_DIR_POS = (20, 90, 240, 140)
CONF_DIR_NEG = (260, 90, 480, 140)
CONF_ROI_BTN = (20, 160, 240, 210)
CONF_FULL_BTN = (260, 160, 480, 210)
CONF_DONE = (20, 230, 480, 290)

_conf = {
    "axis": AXIS,
    "dir": DIR,
    "roi": ROI,
    "drag": False,
    "p0": None,
    "p1": None,
    "roi_select_mode": False,
    "frame_shape": None,
    "done": False,
}


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(img, rect, text, active=False):
    x1, y1, x2, y2 = rect
    color = (0, 255, 0) if active else (255, 255, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1 + 10, y1 + 32), FONT, 0.75, color, 2, cv2.LINE_AA)


def crop_roi(frame, roi):
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    x1 = clamp(int(x1), 0, frame.shape[1] - 1)
    x2 = clamp(int(x2), x1 + 1, frame.shape[1])
    y1 = clamp(int(y1), 0, frame.shape[0] - 1)
    y2 = clamp(int(y2), y1 + 1, frame.shape[0])
    return frame[y1:y2, x1:x2].copy()


def crop_stitch_band(frame):
    if frame is None or frame.size == 0 or not USE_STITCH_BAND:
        return frame
    h, w = frame.shape[:2]
    y1 = int(clamp(round(h * float(STITCH_TOP_FRAC)), 0, h - 2))
    y2 = int(clamp(round(h * float(STITCH_BOTTOM_FRAC)), y1 + 1, h))
    return frame[y1:y2, :].copy()


def resize_keep(frame, scale=1.0):
    if frame is None:
        return None
    if abs(scale - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, (nw, nh), interpolation=interp)


def _roi_norm(p0, p1, w, h):
    x0, y0 = p0
    x1, y1 = p1
    x_a = clamp(min(x0, x1), 0, w - 2)
    x_b = clamp(max(x0, x1), x_a + 1, w - 1)
    y_a = clamp(min(y0, y1), 0, h - 2)
    y_b = clamp(max(y0, y1), y_a + 1, h - 1)
    return int(x_a), int(y_a), int(x_b), int(y_b)


def _axis_dir_label(axis, d):
    if axis == "x":
        return "append=RIGHT" if d > 0 else "append=LEFT"
    return "append=BOTTOM" if d > 0 else "append=TOP"


def _config_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if _conf["roi_select_mode"]:
            _conf["drag"] = True
            _conf["p0"] = (x, y)
            _conf["p1"] = (x, y)
            return

        if in_rect(x, y, CONF_AXIS_X):
            _conf["axis"] = "x"
        elif in_rect(x, y, CONF_AXIS_Y):
            _conf["axis"] = "y"
        elif in_rect(x, y, CONF_DIR_POS):
            _conf["dir"] = +1
        elif in_rect(x, y, CONF_DIR_NEG):
            _conf["dir"] = -1
        elif in_rect(x, y, CONF_ROI_BTN):
            _conf["roi_select_mode"] = True
            _conf["drag"] = False
            _conf["p0"] = None
            _conf["p1"] = None
        elif in_rect(x, y, CONF_FULL_BTN):
            _conf["roi"] = None
            _conf["roi_select_mode"] = False
            _conf["drag"] = False
            _conf["p0"] = None
            _conf["p1"] = None
        elif in_rect(x, y, CONF_DONE):
            _conf["done"] = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if _conf["roi_select_mode"] and _conf["drag"]:
            _conf["p1"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if _conf["roi_select_mode"] and _conf["drag"]:
            _conf["drag"] = False
            if _conf["frame_shape"] is not None and _conf["p0"] is not None and _conf["p1"] is not None:
                h, w = _conf["frame_shape"]
                _conf["roi"] = _roi_norm(_conf["p0"], _conf["p1"], w, h)
            _conf["roi_select_mode"] = False


def first_run_config(get_latest_frame_fn):
    frame = get_latest_frame_fn(wait=True)
    if frame is None:
        raise RuntimeError("Could not read frame for first-run config.")
    h, w = frame.shape[:2]
    _conf["frame_shape"] = (h, w)

    cv2.namedWindow(CONFIG_WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(CONFIG_WIN, _config_mouse)
    cv2.resizeWindow(CONFIG_WIN, w, h)

    while True:
        live = get_latest_frame_fn(wait=False)
        if live is not None:
            frame = live

        ui = frame.copy()

        if _conf["roi"] is not None:
            x1, y1, x2, y2 = _conf["roi"]
            cv2.rectangle(ui, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if _conf["roi_select_mode"] and _conf["p0"] is not None and _conf["p1"] is not None:
            x1, y1, x2, y2 = _roi_norm(_conf["p0"], _conf["p1"], w, h)
            cv2.rectangle(ui, (x1, y1), (x2, y2), (0, 255, 255), 2)

        draw_button(ui, CONF_AXIS_X, "AXIS = X (left/right)", active=(_conf["axis"] == "x"))
        draw_button(ui, CONF_AXIS_Y, "AXIS = Y (top/bottom)", active=(_conf["axis"] == "y"))

        if _conf["axis"] == "x":
            pos_txt = "DIR=+1 => append RIGHT"
            neg_txt = "DIR=-1 => append LEFT"
        else:
            pos_txt = "DIR=+1 => append BOTTOM"
            neg_txt = "DIR=-1 => append TOP"

        draw_button(ui, CONF_DIR_POS, pos_txt, active=(_conf["dir"] == +1))
        draw_button(ui, CONF_DIR_NEG, neg_txt, active=(_conf["dir"] == -1))
        draw_button(ui, CONF_ROI_BTN, "Set ROI (drag on video)", active=_conf["roi_select_mode"])
        draw_button(ui, CONF_FULL_BTN, "Full Frame (ROI=None)", active=(_conf["roi"] is None))
        draw_button(ui, CONF_DONE, "DONE (start scanner)", active=False)

        roi_txt = "ROI=None (full frame)" if _conf["roi"] is None else f"ROI={_conf['roi']}"
        cv2.putText(
            ui,
            f"Selected: axis={_conf['axis']}  {_axis_dir_label(_conf['axis'], _conf['dir'])}  {roi_txt}",
            (20, h - 20),
            FONT,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(CONFIG_WIN, ui)
        key = cv2.waitKey(1) & 0xFF
        if _conf["done"]:
            break
        if key in (27, ord("q")):
            raise SystemExit

    cv2.destroyWindow(CONFIG_WIN)
    return _conf["roi"], _conf["axis"], _conf["dir"]


def feather_blend_1d(a_side, b_side, axis_overlap):
    if axis_overlap <= 0:
        return b_side
    if a_side.ndim != 3 or b_side.ndim != 3 or a_side.shape != b_side.shape:
        return b_side

    h, w = a_side.shape[:2]
    if w == axis_overlap:
        alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    elif h == axis_overlap:
        alpha = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    else:
        return b_side

    a_f = a_side.astype(np.float32)
    b_f = b_side.astype(np.float32)
    out = (1.0 - alpha) * a_f + alpha * b_f
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_strip_axis(pano, strip, axis, direction, overlap_px):
    if pano is None:
        return strip.copy()

    if axis == "x":
        if pano.shape[0] != strip.shape[0]:
            return pano
        overlap = int(clamp(overlap_px, 0, min(pano.shape[1], strip.shape[1])))

        if overlap <= 0:
            return np.concatenate([pano, strip], axis=1) if direction >= 0 else np.concatenate([strip, pano], axis=1)

        if direction >= 0:
            a_side = pano[:, :-overlap]
            pano_edge = pano[:, -overlap:]
            strip_edge = strip[:, :overlap]
            blended = feather_blend_1d(pano_edge, strip_edge, overlap)
            b_side = strip[:, overlap:]
            return np.concatenate([a_side, blended, b_side], axis=1)
        b_side = strip[:, :-overlap]
        strip_edge = strip[:, -overlap:]
        pano_edge = pano[:, :overlap]
        blended = feather_blend_1d(strip_edge, pano_edge, overlap)
        a_side = pano[:, overlap:]
        return np.concatenate([b_side, blended, a_side], axis=1)

    if pano.shape[1] != strip.shape[1]:
        return pano
    overlap = int(clamp(overlap_px, 0, min(pano.shape[0], strip.shape[0])))

    if overlap <= 0:
        return np.concatenate([pano, strip], axis=0) if direction >= 0 else np.concatenate([strip, pano], axis=0)

    if direction >= 0:
        a_side = pano[:-overlap, :]
        pano_edge = pano[-overlap:, :]
        strip_edge = strip[:overlap, :]
        blended = feather_blend_1d(pano_edge, strip_edge, overlap)
        b_side = strip[overlap:, :]
        return np.concatenate([a_side, blended, b_side], axis=0)
    b_side = strip[:-overlap, :]
    strip_edge = strip[-overlap:, :]
    pano_edge = pano[:overlap, :]
    blended = feather_blend_1d(strip_edge, pano_edge, overlap)
    a_side = pano[overlap:, :]
    return np.concatenate([b_side, blended, a_side], axis=0)


def motion_metrics_near_only(prev_gray, cur_gray, diff_thr, use_bottom_frac=0.4, ignore_top_px=0):
    if prev_gray is None or cur_gray is None:
        return 0.0, 0.0, (0, 0)

    h = cur_gray.shape[0]
    y0 = int(h * (1.0 - float(clamp(use_bottom_frac, 0.05, 1.0))))
    y0 = clamp(y0, 0, h - 1)

    diff = cv2.absdiff(cur_gray, prev_gray)
    if ignore_top_px > 0 and ignore_top_px < diff.shape[0]:
        diff[:ignore_top_px, :] = 0
    if y0 > 0:
        diff[:y0, :] = 0

    _, th = cv2.threshold(diff, diff_thr, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)

    changed = float(np.count_nonzero(th))
    total = float(th.size)
    frac = changed / max(1.0, total)
    meanv = float(np.mean(diff))
    return frac, meanv, (y0, h)


def try_init_cuda(use_gpu, device_id=0):
    if not use_gpu:
        return False, "CPU mode (USE_GPU=False)"
    try:
        if not hasattr(cv2, "cuda"):
            return False, "CPU mode (OpenCV built without CUDA module)"
        cnt = int(cv2.cuda.getCudaEnabledDeviceCount())
        if cnt <= 0:
            return False, "CPU mode (no CUDA devices detected by OpenCV)"
        device_id = int(clamp(device_id, 0, cnt - 1))
        cv2.cuda.setDevice(device_id)
        _ = cv2.cuda_GpuMat()
        return True, f"GPU mode (CUDA) device={device_id} count={cnt}"
    except Exception as e:
        return False, f"CPU mode (CUDA init failed: {e})"


def motion_preprocess(bgr, ds_w, blur_k, use_gpu=False):
    h, w = bgr.shape[:2]
    ds_w = clamp(int(ds_w), 80, 1920)
    ds_h = max(1, int(h * (ds_w / float(w))))

    if use_gpu:
        g = cv2.cuda_GpuMat()
        g.upload(bgr)
        g_small = cv2.cuda.resize(g, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        g_gray = cv2.cuda.cvtColor(g_small, cv2.COLOR_BGR2GRAY)
        if blur_k and blur_k >= 3 and blur_k % 2 == 1:
            g_gray = cv2.cuda.GaussianBlur(g_gray, (blur_k, blur_k), 0)
        return g_gray.download()

    small = cv2.resize(bgr, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if blur_k and blur_k >= 3 and blur_k % 2 == 1:
        g = cv2.GaussianBlur(g, (blur_k, blur_k), 0)
    return g


def make_feature_band_mask(gray, band_frac=0.5, ignore_top_px=0):
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * (1.0 - float(clamp(band_frac, 0.05, 1.0))))
    y0 = clamp(y0, 0, h - 1)
    if ignore_top_px > 0:
        y0 = max(y0, int(ignore_top_px))
    mask[y0:h, :] = 255
    return mask


def robust_median_shift(vals, mad_thr):
    if len(vals) == 0:
        return None, 0
    arr = np.asarray(vals, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-6:
        return med, len(arr)
    good = np.abs(arr - med) <= (mad_thr + 2.5 * mad)
    arr2 = arr[good]
    if len(arr2) == 0:
        return None, 0
    return float(np.median(arr2)), int(len(arr2))


def estimate_translation_orb(prev_bgr, cur_bgr, axis="x"):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    mask_prev = make_feature_band_mask(prev_gray, FEATURE_BAND_FRAC, FEATURE_IGNORE_TOP_PX)
    mask_cur = make_feature_band_mask(cur_gray, FEATURE_BAND_FRAC, FEATURE_IGNORE_TOP_PX)

    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_NLEVELS)

    kp1, des1 = orb.detectAndCompute(prev_gray, mask_prev)
    kp2, des2 = orb.detectAndCompute(cur_gray, mask_cur)

    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return None

    if len(kp1) < MIN_GOOD_MATCHES or len(kp2) < MIN_GOOD_MATCHES:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < MATCH_RATIO_TEST * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return None

    dxs = []
    dys = []
    for m in good:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        dxs.append(float(p2[0] - p1[0]))
        dys.append(float(p2[1] - p1[1]))

    dx_med, dx_inliers = robust_median_shift(dxs, MAD_OUTLIER_THR)
    dy_med, dy_inliers = robust_median_shift(dys, MAD_OUTLIER_THR)

    if dx_med is None or dy_med is None:
        return None

    if axis == "x":
        if abs(dx_med) > MAX_MATCH_STEP_X:
            return None
    elif abs(dy_med) > MAX_MATCH_STEP_Y:
        return None

    return {
        "dx": dx_med,
        "dy": dy_med,
        "matches": len(good),
        "dx_inliers": dx_inliers,
        "dy_inliers": dy_inliers,
    }


def estimate_translation_phase(prev_bgr, cur_bgr, axis="x"):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)

    mask = make_feature_band_mask(prev_gray, FEATURE_BAND_FRAC, FEATURE_IGNORE_TOP_PX)
    p = prev_gray.astype(np.float32)
    c = cur_gray.astype(np.float32)
    p[mask == 0] = 0.0
    c[mask == 0] = 0.0

    shift, response = cv2.phaseCorrelate(p, c)
    dx, dy = float(shift[0]), float(shift[1])
    response = float(response)
    if response < MIN_PHASE_RESPONSE:
        return None

    if axis == "x" and abs(dx) > MAX_MATCH_STEP_X:
        return None
    if axis == "y" and abs(dy) > MAX_MATCH_STEP_Y:
        return None

    return {
        "dx": dx,
        "dy": dy,
        "matches": 0,
        "dx_inliers": 0,
        "dy_inliers": 0,
        "phase_response": response,
    }


def extract_feature_matched_strip(roi_bgr, axis, signed_shift, slit_x, slit_y):
    h, w = roi_bgr.shape[:2]

    if axis == "x":
        step = int(round(abs(signed_shift)))
        step = clamp(step, MIN_FEATURE_STEP, min(MAX_MATCH_STEP_X, max(2, w // 2)))

        if signed_shift >= 0:
            x2 = clamp(slit_x, 1, w)
            x1 = clamp(x2 - step, 0, w - 1)
            if x2 <= x1:
                return None, 0
            strip = roi_bgr[:, x1:x2].copy()
        else:
            x1 = clamp(slit_x, 0, w - 1)
            x2 = clamp(x1 + step, x1 + 1, w)
            if x2 <= x1:
                return None, 0
            strip = roi_bgr[:, x1:x2].copy()

        return strip, step

    step = int(round(abs(signed_shift)))
    step = clamp(step, MIN_FEATURE_STEP, min(MAX_MATCH_STEP_Y, max(2, h // 2)))

    if signed_shift >= 0:
        y2 = clamp(slit_y, 1, h)
        y1 = clamp(y2 - step, 0, h - 1)
        if y2 <= y1:
            return None, 0
        strip = roi_bgr[y1:y2, :].copy()
    else:
        y1 = clamp(slit_y, 0, h - 1)
        y2 = clamp(y1 + step, y1 + 1, h)
        if y2 <= y1:
            return None, 0
        strip = roi_bgr[y1:y2, :].copy()

    return strip, step


class FrameGrabber:
    def __init__(self, src, max_queue=0, cap_buffersize=16):
        self.src = src
        self.max_queue = int(max_queue)
        self.cap_buffersize = int(cap_buffersize)

        self.cap = None
        self.q = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.read_fail_count = 0
        self.total_frames = 0

        self._open_capture()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open: {self.src}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffersize)
        except Exception:
            pass

    def _reopen_capture(self):
        try:
            print("🔄 Reopening stream...")
            self._open_capture()
            self.read_fail_count = 0
        except Exception as e:
            print(f"⚠ Reopen failed: {e}")

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.read_fail_count += 1
                time.sleep(READ_FAIL_RETRY_SEC)

                if REOPEN_STREAM_ON_FAIL and self.read_fail_count >= MAX_CONSECUTIVE_READ_FAILS:
                    self._reopen_capture()
                continue

            self.read_fail_count = 0
            ts = time.time()
            self.total_frames += 1

            with self.cond:
                if self.max_queue > 0:
                    while self.running and len(self.q) >= self.max_queue:
                        self.cond.wait(timeout=0.05)

                if not self.running:
                    break

                self.q.append((ts, frame))
                self.cond.notify_all()

    def get(self, wait=True, timeout=1.0):
        with self.cond:
            if not wait:
                if not self.q:
                    return None
            else:
                end = time.time() + float(timeout)
                while self.running and not self.q and time.time() < end:
                    self.cond.wait(timeout=0.05)
                if not self.q:
                    return None

            item = self.q.popleft()
            self.cond.notify_all()
            return item

    def latest_frame(self, wait=True):
        item = self.get(wait=wait, timeout=1.0)
        return None if item is None else item[1]

    def qsize(self):
        with self.lock:
            return len(self.q)

    def release(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()


state = {
    "capturing": False,
    "auto": True,
    "pano": None,
    "frames_used": 0,
    "strip_ema": None,
    "axis": AXIS,
    "dir": DIR,
    "prev_mgray": None,
    "last_motion_t": None,
    "motion_on_t": None,
    "last_strip_w": FIXED_STRIP_W,
    "last_strip_h": FIXED_STRIP_H,
    "prev_capture_roi": None,
    "last_match_dx": 0.0,
    "last_match_dy": 0.0,
    "last_match_step": 0,
    "last_match_count": 0,
    "last_match_ok": False,
    "last_match_src": "none",
    "step_ema": None,
    "pre_roll": deque(maxlen=AUTO_PREROLL_FRAMES),
    "postroll_left": 0,
    "captured_frames": 0,
    "stitched_frames": 0,
    "residual_shift": 0.0,
    "capture_started_t": None,
    "strips_added": 0,
    "smoothed_dx": 0.0,
    "frame_idx": 0,
    "stop_reason": "",
}


def reset_panorama():
    state["pano"] = None
    state["frames_used"] = 0
    state["strip_ema"] = None
    state["prev_mgray"] = None
    state["last_strip_w"] = FIXED_STRIP_W
    state["last_strip_h"] = FIXED_STRIP_H
    state["prev_capture_roi"] = None
    state["last_match_dx"] = 0.0
    state["last_match_dy"] = 0.0
    state["last_match_step"] = 0
    state["last_match_count"] = 0
    state["last_match_ok"] = False
    state["last_match_src"] = "none"
    state["step_ema"] = None
    state["pre_roll"].clear()
    state["postroll_left"] = 0
    state["captured_frames"] = 0
    state["stitched_frames"] = 0
    state["residual_shift"] = 0.0
    state["capture_started_t"] = None
    state["strips_added"] = 0
    state["smoothed_dx"] = 0.0
    state["frame_idx"] = 0
    state["stop_reason"] = ""


def postprocess_panorama_for_save(pano):
    if pano is None:
        return None
    out = pano.copy()
    if SAVE_SCALE != 1.0:
        out = resize_keep(out, SAVE_SCALE)
    if any(v > 0 for v in [FINAL_PAD_TOP, FINAL_PAD_BOTTOM, FINAL_PAD_LEFT, FINAL_PAD_RIGHT]):
        out = cv2.copyMakeBorder(
            out,
            FINAL_PAD_TOP,
            FINAL_PAD_BOTTOM,
            FINAL_PAD_LEFT,
            FINAL_PAD_RIGHT,
            borderType=cv2.BORDER_CONSTANT,
            value=(20, 20, 20),
        )
    return out


def save_panorama(tag="manual"):
    pano = state["pano"]
    if pano is None:
        print("⚠ Nothing to save (panorama empty).")
        return False

    pano_to_save = postprocess_panorama_for_save(pano)
    if pano_to_save is None:
        return False

    if pano_to_save.shape[1] < MIN_SAVE_W or pano_to_save.shape[0] < MIN_SAVE_H:
        print(f"⚠ Skip save (too small) size={pano_to_save.shape[1]}x{pano_to_save.shape[0]}")
        return False

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_DIR, f"scan_{tag}_{state['axis']}_{'pos' if state['dir'] > 0 else 'neg'}_{ts}.png")
    ok = cv2.imwrite(out, pano_to_save)
    if ok:
        print(
            f"✅ Saved: {out} | raw={pano.shape[1]}x{pano.shape[0]} "
            f"| saved={pano_to_save.shape[1]}x{pano_to_save.shape[0]} "
            f"| frames_used={state['frames_used']}"
        )
    else:
        print(f"❌ Failed to save: {out}")
    return ok


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if in_rect(x, y, BTN_START):
        state["auto"] = False
        state["capturing"] = True
        print("▶ START capture (MANUAL)")
    elif in_rect(x, y, BTN_STOP):
        state["auto"] = False
        state["capturing"] = False
        print("⏹ STOP capture (MANUAL)")
    elif in_rect(x, y, BTN_SAVE):
        save_panorama(tag="manual")
    elif in_rect(x, y, BTN_RESET):
        reset_panorama()
        print("♻ RESET panorama")


def append_seed_strip_from_frame(roi_bgr, slit_x, slit_y):
    if state["axis"] == "x":
        step = int(FIXED_STRIP_W)
        step = clamp(step, 1, max(1, roi_bgr.shape[1] // 2))
        xs = clamp(slit_x - step // 2, 0, roi_bgr.shape[1] - 1)
        xe = clamp(xs + step, xs + 1, roi_bgr.shape[1])
        strip = roi_bgr[:, xs:xe].copy()
        state["last_strip_w"] = strip.shape[1]
    else:
        step = int(FIXED_STRIP_H)
        step = clamp(step, 1, max(1, roi_bgr.shape[0] // 2))
        ys = clamp(slit_y - step // 2, 0, roi_bgr.shape[0] - 1)
        ye = clamp(ys + step, ys + 1, roi_bgr.shape[0])
        strip = roi_bgr[ys:ye, :].copy()
        state["last_strip_h"] = strip.shape[0]

    if strip.size <= 0:
        return

    state["pano"] = paste_strip_axis(state["pano"], strip, state["axis"], state["dir"], 0)
    state["frames_used"] += 1
    state["stitched_frames"] += 1


def append_feature_matched_strip_from_frame(roi_bgr, slit_x, slit_y):
    prev_roi = state["prev_capture_roi"]

    if prev_roi is None:
        append_seed_strip_from_frame(roi_bgr, slit_x, slit_y)
        state["prev_capture_roi"] = roi_bgr.copy()
        state["last_match_step"] = 0
        state["last_match_count"] = 0
        state["last_match_dx"] = 0.0
        state["last_match_dy"] = 0.0
        state["last_match_ok"] = False
        return 0

    orb_match = None
    phase_match = None
    match = None
    if USE_FEATURE_MATCHING:
        orb_match = estimate_translation_orb(prev_roi, roi_bgr, axis=state["axis"])
    if USE_PHASE_CORR:
        phase_match = estimate_translation_phase(prev_roi, roi_bgr, axis=state["axis"])

    if orb_match is not None and phase_match is not None:
        orb_shift = orb_match["dx"] if state["axis"] == "x" else orb_match["dy"]
        phase_shift = phase_match["dx"] if state["axis"] == "x" else phase_match["dy"]
        # On repetitive ribbed containers ORB can underestimate motion; prefer phase when disagreeing strongly.
        match = phase_match if abs(orb_shift - phase_shift) > 8.0 else orb_match
    else:
        match = orb_match if orb_match is not None else phase_match

    if match is not None:
        signed_shift = match["dx"] if state["axis"] == "x" else match["dy"]

        step_abs = abs(float(signed_shift))
        if state["step_ema"] is None:
            state["step_ema"] = step_abs
        else:
            a = float(clamp(STEP_EMA_ALPHA, 0.0, 0.95))
            state["step_ema"] = a * state["step_ema"] + (1.0 - a) * step_abs

        signed_for_extract = np.sign(signed_shift) * max(float(MIN_FEATURE_STEP), float(state["step_ema"])) * float(STEP_SCALE)
        strip, used_step = extract_feature_matched_strip(roi_bgr, state["axis"], signed_for_extract, slit_x, slit_y)
        used_step = int(clamp(used_step, 0, MAX_EFFECTIVE_STEP))

        state["last_match_dx"] = float(match["dx"])
        state["last_match_dy"] = float(match["dy"])
        state["last_match_count"] = int(match["matches"])
        state["last_match_step"] = int(used_step)
        state["last_match_ok"] = strip is not None and used_step >= max(MIN_FEATURE_STEP, MIN_EFFECTIVE_STEP)
        state["last_match_src"] = "phase" if "phase_response" in match else "orb"

        if strip is not None and strip.size > 0:
            if state["axis"] == "x":
                state["last_strip_w"] = strip.shape[1]
            else:
                state["last_strip_h"] = strip.shape[0]

            state["pano"] = paste_strip_axis(state["pano"], strip, state["axis"], state["dir"], 0)
            state["frames_used"] += 1
            state["stitched_frames"] += 1

        state["prev_capture_roi"] = roi_bgr.copy()
        return used_step

    state["last_match_ok"] = False
    state["last_match_count"] = 0
    state["last_match_dx"] = 0.0
    state["last_match_dy"] = 0.0
    state["last_match_step"] = 0
    state["last_match_src"] = "none"

    if FALLBACK_TO_FIXED_IF_NO_MATCH:
        append_seed_strip_from_frame(roi_bgr, slit_x, slit_y)

    state["prev_capture_roi"] = roi_bgr.copy()
    return 0


def append_fixed_strip_from_frame(roi_bgr, slit_x, slit_y):
    if state["axis"] == "x":
        strip_w = clamp(int(STRIP_W), 1, max(1, roi_bgr.shape[1] // 2))
        xs = clamp(slit_x - strip_w // 2, 0, roi_bgr.shape[1] - 1)
        xe = clamp(xs + strip_w, xs + 1, roi_bgr.shape[1])
        strip = roi_bgr[:, xs:xe].copy()
        state["last_strip_w"] = strip.shape[1]
    else:
        strip_h = clamp(int(STRIP_H), 1, max(1, roi_bgr.shape[0] // 2))
        ys = clamp(slit_y - strip_h // 2, 0, roi_bgr.shape[0] - 1)
        ye = clamp(ys + strip_h, ys + 1, roi_bgr.shape[0])
        strip = roi_bgr[ys:ye, :].copy()
        state["last_strip_h"] = strip.shape[0]

    if strip.size <= 0:
        return 0

    state["pano"] = paste_strip_axis(state["pano"], strip, state["axis"], state["dir"], SEAM_OVERLAP_PX)
    state["frames_used"] += 1
    state["stitched_frames"] += 1
    state["strips_added"] += 1
    return int(strip.shape[1] if state["axis"] == "x" else strip.shape[0])


def append_smart_non_overlap_strip_from_frame(roi_bgr, slit_x, slit_y):
    prev_roi = state["prev_capture_roi"]
    if prev_roi is None:
        append_fixed_strip_from_frame(roi_bgr, slit_x, slit_y)
        state["prev_capture_roi"] = roi_bgr.copy()
        state["residual_shift"] = 0.0
        return 0

    # Phase correlation is stable on repetitive container ribs for translational motion.
    match = estimate_translation_phase(prev_roi, roi_bgr, axis=state["axis"])
    if match is None:
        append_fixed_strip_from_frame(roi_bgr, slit_x, slit_y)
        state["prev_capture_roi"] = roi_bgr.copy()
        return 0

    signed_shift = float(match["dx"] if state["axis"] == "x" else match["dy"])
    state["last_match_dx"] = float(match["dx"])
    state["last_match_dy"] = float(match["dy"])
    # Smooth dx to reduce jitter and preserve proportions.
    state["smoothed_dx"] = 0.75 * float(state["smoothed_dx"]) + 0.25 * signed_shift
    signed_shift = float(state["smoothed_dx"])
    if AUTO_DIRECTION_FROM_DX and abs(signed_shift) >= 1.0:
        # In screen coordinates, positive dx means object moved right,
        # so newly revealed content arrives from left side.
        state["dir"] = -1 if signed_shift >= 0 else 1

    state["residual_shift"] += signed_shift
    step = int(abs(state["residual_shift"]))
    step = int(clamp(step, 0, MAX_APPEND_STEP))
    if step < int(MIN_APPEND_STEP):
        state["last_match_step"] = int(step)
        state["prev_capture_roi"] = roi_bgr.copy()
        return 0

    signed_for_extract = float(np.sign(state["residual_shift"])) * float(step)
    strip, used_step = extract_feature_matched_strip(roi_bgr, state["axis"], signed_for_extract, slit_x, slit_y)
    if strip is not None and strip.size > 0:
        state["last_match_step"] = int(used_step)
        state["last_match_ok"] = True
        state["last_match_src"] = "smart_phase"
        state["pano"] = paste_strip_axis(state["pano"], strip, state["axis"], state["dir"], 0)
        state["frames_used"] += 1
        state["stitched_frames"] += 1
        state["strips_added"] += 1
        state["residual_shift"] -= float(np.sign(state["residual_shift"])) * float(used_step)

    state["prev_capture_roi"] = roi_bgr.copy()
    return int(used_step if strip is not None else 0)


def start_capture_with_preroll(slit_x, slit_y):
    state["capturing"] = True
    state["prev_capture_roi"] = None
    state["step_ema"] = None
    pre = list(state["pre_roll"])
    for frm in pre:
        h, w = frm.shape[:2]
        sx = clamp(int(slit_x), 1, w - 2)
        sy = clamp(int(slit_y), 1, h - 2)
        if USE_SMART_NON_OVERLAP:
            append_smart_non_overlap_strip_from_frame(frm, sx, sy)
        elif USE_FIXED_STRIP_ONLY:
            append_fixed_strip_from_frame(frm, sx, sy)
        else:
            append_feature_matched_strip_from_frame(frm, sx, sy)
    state["pre_roll"].clear()


def main():
    gpu_ok, gpu_msg = try_init_cuda(USE_GPU, GPU_DEVICE_ID)
    print(f"🖥️ Compute: {gpu_msg}")

    grabber = FrameGrabber(INPUT, max_queue=FRAMEBUFFER_MAX, cap_buffersize=CAP_BUFFERSIZE)

    global ROI
    ROI, state["axis"], state["dir"] = first_run_config(get_latest_frame_fn=lambda wait: grabber.latest_frame(wait=wait))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    ui_period = 1.0 / max(1e-6, TARGET_FPS)
    last_ui = 0.0

    print("📌 Ready.")
    print("Starts immediately on motion.")
    print("Stops after 2.0 seconds of no motion.")
    print("Uses ORB feature matching to estimate new strip width and reduce overlap.")
    print("Unlimited FIFO queue by default; no intentional frame dropping.")
    print("⌨ Hotkeys: A auto | SPACE manual toggle | D direction | X/Y axis | S save | R reset | G gpu | Q quit")

    while True:
        state["frame_idx"] += 1
        item = grabber.get(wait=True, timeout=2.0)
        if item is None:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            continue

        now = time.time()
        _, frame = item

        roi_bgr = crop_roi(frame, ROI)
        h, w = roi_bgr.shape[:2]
        if w < 10 or h < 10:
            continue

        stitch_bgr = crop_stitch_band(roi_bgr)
        if stitch_bgr is None or stitch_bgr.size == 0:
            continue
        state["pre_roll"].append(stitch_bgr.copy())

        mgray = motion_preprocess(roi_bgr, ds_w=MOTION_DS_W, blur_k=MOTION_BLUR, use_gpu=gpu_ok)

        frac, meanv, _ = motion_metrics_near_only(
            state["prev_mgray"],
            mgray,
            diff_thr=MOTION_DIFF_THR,
            use_bottom_frac=MOTION_USE_BOTTOM_FRAC,
            ignore_top_px=IGNORE_TOP_PX,
        )

        state["prev_mgray"] = mgray
        is_motion = frac >= float(MOTION_AREA_FRAC)

        if is_motion:
            state["last_motion_t"] = now
            if state["motion_on_t"] is None:
                state["motion_on_t"] = now
        else:
            state["motion_on_t"] = None

        if state["auto"]:
            if not state["capturing"]:
                if is_motion:
                    start_capture_with_preroll((w // 2) if SLIT_X < 0 else int(SLIT_X), (h // 2) if SLIT_Y < 0 else int(SLIT_Y))
                    state["postroll_left"] = 0
                    state["capture_started_t"] = now
                    state["stop_reason"] = ""
                    print("▶ AUTO START")
            else:
                last_m = state["last_motion_t"]
                cap_age = 0.0 if state["capture_started_t"] is None else (now - state["capture_started_t"])
                if last_m is not None and (now - last_m) >= NO_MOTION_STOP_SEC and cap_age >= MIN_CAPTURE_SEC_BEFORE_STOP_CHECK:
                    if state["postroll_left"] <= 0:
                        state["postroll_left"] = int(AUTO_POSTROLL_FRAMES)
                    else:
                        state["postroll_left"] -= 1
                        if state["postroll_left"] <= 0:
                            state["capturing"] = False
                            state["stop_reason"] = "lane_empty_timeout"
                            print("⏹ AUTO STOP -> SAVE + RESET")
                            save_panorama(tag="auto")
                            reset_panorama()
                            state["motion_on_t"] = None
                            state["last_motion_t"] = None
                            continue
                else:
                    state["postroll_left"] = 0

        sh, sw = stitch_bgr.shape[:2]
        slit_x = (sw // 2) if SLIT_X < 0 else int(SLIT_X)
        slit_y = (sh // 2) if SLIT_Y < 0 else int(SLIT_Y)
        slit_x = clamp(slit_x, 1, sw - 2)
        slit_y = clamp(slit_y, 1, sh - 2)

        if state["capturing"] and USE_EVERY_FRAME_STITCH:
            state["captured_frames"] += 1
            if USE_SMART_NON_OVERLAP:
                append_smart_non_overlap_strip_from_frame(stitch_bgr, slit_x, slit_y)
            elif USE_FIXED_STRIP_ONLY:
                append_fixed_strip_from_frame(stitch_bgr, slit_x, slit_y)
            else:
                append_feature_matched_strip_from_frame(stitch_bgr, slit_x, slit_y)
        else:
            state["prev_capture_roi"] = None
            state["step_ema"] = None
            state["residual_shift"] = 0.0

        if SHOW_DEBUG and (state["frame_idx"] % int(max(1, DEBUG_LOG_EVERY_N_FRAMES)) == 0):
            dx_dbg = state["last_match_dx"] if state["axis"] == "x" else state["last_match_dy"]
            print(
                f"DBG motion_detected={is_motion} dx={dx_dbg:.2f} smoothed_dx={state['smoothed_dx']:.2f} "
                f"strip_width={state['last_match_step']} direction={state['dir']} frames_used={state['frames_used']} "
                f"strips_added={state['strips_added']} stop_reason={state['stop_reason']}"
            )

        if now - last_ui >= ui_period:
            last_ui = now
            disp = roi_bgr.copy()

            band_y0 = int(h * (1.0 - float(clamp(MOTION_USE_BOTTOM_FRAC, 0.05, 1.0))))
            band_y0 = clamp(band_y0, 0, h - 1)
            cv2.rectangle(disp, (0, band_y0), (w - 1, h - 1), (0, 255, 0), 2)

            if state["axis"] == "x":
                sw = max(1, int(state["last_strip_w"]))
                xs = clamp(slit_x - sw // 2, 0, w - 1)
                xe = clamp(xs + sw, xs + 1, w)
                cv2.line(disp, (slit_x, 0), (slit_x, h - 1), (0, 255, 255), 2)
                cv2.rectangle(disp, (xs, 0), (xe, h - 1), (255, 255, 0), 2)
            else:
                sh = max(1, int(state["last_strip_h"]))
                ys = clamp(slit_y - sh // 2, 0, h - 1)
                ye = clamp(ys + sh, ys + 1, h)
                cv2.line(disp, (0, slit_y), (w - 1, slit_y), (0, 255, 255), 2)
                cv2.rectangle(disp, (0, ys), (w - 1, ye), (255, 255, 0), 2)

            draw_button(disp, BTN_START, "START", active=(not state["auto"] and state["capturing"]))
            draw_button(disp, BTN_STOP, "STOP", active=(not state["auto"] and (not state["capturing"])))
            draw_button(disp, BTN_SAVE, "SAVE", active=False)
            draw_button(disp, BTN_RESET, "RESET", active=False)

            pano_w = 0 if state["pano"] is None else state["pano"].shape[1]
            pano_h = 0 if state["pano"] is None else state["pano"].shape[0]
            cur_strip = state["last_strip_w"] if state["axis"] == "x" else state["last_strip_h"]
            mode_txt = "AUTO" if state["auto"] else "MANUAL"
            dir_txt = _axis_dir_label(state["axis"], state["dir"])
            compute_txt = "GPU" if gpu_ok else "CPU"
            qlen = grabber.qsize()

            cv2.putText(
                disp,
                f"{compute_txt} {mode_txt} axis={state['axis']} {dir_txt} strip={cur_strip}px pano={pano_w}x{pano_h} stitched={state['stitched_frames']}/{state['captured_frames']} q={qlen}",
                (10, 90),
                FONT,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if SHOW_DEBUG:
                cv2.putText(
                    disp,
                    f"motion={frac:.4f} mean={meanv:.2f} total_frames={grabber.total_frames}",
                    (10, 120),
                    FONT,
                    0.54,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if state["capturing"] and state["last_motion_t"] is not None:
                    remain = max(0.0, NO_MOTION_STOP_SEC - (now - state["last_motion_t"]))
                else:
                    remain = NO_MOTION_STOP_SEC
                cv2.putText(
                    disp,
                    f"auto save+reset in: {remain:.1f}s",
                    (10, 145),
                    FONT,
                    0.54,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    disp,
                    f"stitch={'smart_non_overlap' if USE_SMART_NON_OVERLAP else ('fixed' if USE_FIXED_STRIP_ONLY else state['last_match_src'])} ok={state['last_match_ok']} matches={state['last_match_count']} dx={state['last_match_dx']:.1f} dy={state['last_match_dy']:.1f} step={state['last_match_step']} resid={state['residual_shift']:.2f}",
                    (10, 170),
                    FONT,
                    0.54,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("s"):
            save_panorama(tag="manual")
        elif key == ord("r"):
            reset_panorama()
            print("♻ RESET panorama")
        elif key == ord(" "):
            state["auto"] = False
            state["capturing"] = not state["capturing"]
            if state["capturing"]:
                state["prev_capture_roi"] = None
                state["step_ema"] = None
            print(f"{'▶' if state['capturing'] else '⏹'} MANUAL toggle")
        elif key in (ord("a"), ord("A")):
            state["auto"] = not state["auto"]
            print(f"🧠 AUTO: {'ON' if state['auto'] else 'OFF'}")
        elif key in (ord("x"), ord("X")):
            state["axis"] = "x"
            print("🧭 axis=x")
        elif key in (ord("y"), ord("Y")):
            state["axis"] = "y"
            print("🧭 axis=y")
        elif key in (ord("d"), ord("D")):
            state["dir"] *= -1
            print(f"🔁 {_axis_dir_label(state['axis'], state['dir'])}")
        elif key in (ord("g"), ord("G")):
            if gpu_ok:
                gpu_ok = False
                print("🖥️ Switched to CPU")
            else:
                ok2, msg2 = try_init_cuda(True, GPU_DEVICE_ID)
                gpu_ok = ok2
                print(f"🖥️ {msg2}")

    grabber.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
