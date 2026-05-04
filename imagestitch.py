import cv2
import numpy as np
import os
import time
import threading
from collections import deque

# =========================
# CONFIG
# =========================
RTSP_URL = "rtsp://admin:admin@10.19.223.17/profile2/media.smp"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional ROI for processing (x1, y1, x2, y2). None = full frame
ROI = None

# Direction hint:
#  +1 = append right (object moving left->right)
#  -1 = append left  (object moving right->left)
#  0  = auto from dx sign
DIRECTION = 0

TARGET_FPS = 25
MOTION_THRESHOLD = 16
PRESENCE_FRAC_THRESHOLD = 0.015
MOTION_FRAC_THRESHOLD = 0.002
NO_MOTION_STOP_SEC = 2.0
PRE_BUFFER_SEC = 0.8

MIN_STRIP_WIDTH = 2
MAX_STRIP_WIDTH = 140
SEAM_BLEND_PX = 8
PREVIEW_SCALE = 1.0
DEBUG_DISPLAY = True

SHARPNESS_FILTER_ON = True
MIN_SHARPNESS_VAR = 22.0

CAPTURE_REF_ON_START = True
REFERENCE_IMAGE_PATH = "empty_lane_reference.png"

FRAME_QUEUE_MAX = 800
CAP_PROP_BUFFERSIZE = 128

WINDOW = "TruckSideProfileStitch"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# UTIL
# =========================
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def crop_roi(frame, roi):
    if roi is None:
        return frame
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    x1 = int(clamp(x1, 0, w - 1))
    x2 = int(clamp(x2, x1 + 1, w))
    y1 = int(clamp(y1, 0, h - 1))
    y2 = int(clamp(y2, y1 + 1, h))
    return frame[y1:y2, x1:x2].copy()


def resize_keep(frame, scale):
    if abs(scale - 1.0) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)


def laplacian_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def feather_blend_vertical(a_right, b_left, blend_w):
    if blend_w <= 0:
        return b_left
    blend_w = int(clamp(blend_w, 1, min(a_right.shape[1], b_left.shape[1])))
    ar = a_right[:, -blend_w:].astype(np.float32)
    bl = b_left[:, :blend_w].astype(np.float32)
    alpha = np.linspace(0.0, 1.0, blend_w, dtype=np.float32)[None, :, None]
    out = (1.0 - alpha) * ar + alpha * bl
    return np.clip(out, 0, 255).astype(np.uint8)


def append_strip(pano, strip, append_dir, blend_px):
    if pano is None:
        return strip.copy()
    if pano.shape[0] != strip.shape[0]:
        return pano

    blend = int(clamp(blend_px, 0, min(pano.shape[1], strip.shape[1])))
    if blend == 0:
        return np.concatenate([pano, strip], axis=1) if append_dir >= 0 else np.concatenate([strip, pano], axis=1)

    if append_dir >= 0:
        left = pano[:, :-blend]
        seam = feather_blend_vertical(pano, strip, blend)
        right = strip[:, blend:]
        return np.concatenate([left, seam, right], axis=1)

    left = strip[:, :-blend]
    seam = feather_blend_vertical(strip, pano, blend)
    right = pano[:, blend:]
    return np.concatenate([left, seam, right], axis=1)


# =========================
# FRAME GRABBER
# =========================
class FrameGrabber:
    def __init__(self, src, queue_max=500, cap_buf=64):
        self.src = src
        self.queue_max = int(queue_max)
        self.cap_buf = int(cap_buf)
        self.q = deque()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.total_received = 0

        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.src}")

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buf)
        except Exception:
            pass

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            ts = time.time()
            with self.cond:
                self.total_received += 1
                self.q.append((ts, frame))
                if self.queue_max > 0:
                    while len(self.q) > self.queue_max:
                        self.q.popleft()
                self.cond.notify_all()

    def get(self, timeout=1.0):
        with self.cond:
            end = time.time() + float(timeout)
            while self.running and not self.q and time.time() < end:
                self.cond.wait(timeout=0.02)
            if not self.q:
                return None
            return self.q.popleft()

    def latest_frame(self, timeout=1.0):
        item = self.get(timeout=timeout)
        return None if item is None else item[1]

    def qsize(self):
        with self.lock:
            return len(self.q)

    def release(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


# =========================
# REFERENCE / METRICS
# =========================
def load_or_capture_reference(grabber):
    if os.path.exists(REFERENCE_IMAGE_PATH):
        ref = cv2.imread(REFERENCE_IMAGE_PATH)
        if ref is not None:
            return ref

    frame = grabber.latest_frame(timeout=2.0)
    if frame is None:
        raise RuntimeError("Could not capture reference frame.")
    if CAPTURE_REF_ON_START:
        cv2.imwrite(REFERENCE_IMAGE_PATH, frame)
    return frame


def compute_presence_motion(ref_gray, prev_gray, cur_gray):
    if ref_gray is None or cur_gray is None:
        return 0.0, 0.0

    ref_diff = cv2.absdiff(cur_gray, ref_gray)
    _, ref_th = cv2.threshold(ref_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    ref_th = cv2.medianBlur(ref_th, 5)
    presence_frac = float(np.count_nonzero(ref_th)) / float(ref_th.size)

    if prev_gray is None:
        return presence_frac, 0.0

    mov_diff = cv2.absdiff(cur_gray, prev_gray)
    _, mov_th = cv2.threshold(mov_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    mov_th = cv2.medianBlur(mov_th, 5)
    motion_frac = float(np.count_nonzero(mov_th)) / float(mov_th.size)
    return presence_frac, motion_frac


def estimate_dx_phase(prev_gray, cur_gray):
    p = prev_gray.astype(np.float32)
    c = cur_gray.astype(np.float32)
    shift, response = cv2.phaseCorrelate(p, c)
    return float(shift[0]), float(response)


# =========================
# MAIN APP
# =========================
def main():
    grabber = FrameGrabber(RTSP_URL, queue_max=FRAME_QUEUE_MAX, cap_buf=CAP_PROP_BUFFERSIZE)

    ref_full = load_or_capture_reference(grabber)
    ref_roi = crop_roi(ref_full, ROI)
    ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)

    pre_buffer = deque()
    pre_buffer_max = max(2, int(round(PRE_BUFFER_SEC * max(1, TARGET_FPS))))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    capturing = False
    pano = None
    prev_gray = None
    prev_roi = None
    residual_dx = 0.0
    smoothed_dx = 0.0
    last_presence_t = None
    last_motion_t = None
    capture_started_t = None
    stop_reason = ""

    # stats
    frames_processed = 0
    frames_used = 0
    strips_added = 0
    strip_sum = 0

    while True:
        item = grabber.get(timeout=2.0)
        if item is None:
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break
            continue

        ts, frame = item
        roi = crop_roi(frame, ROI)
        h, w = roi.shape[:2]
        if w < 20 or h < 20:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)
        is_sharp = (not SHARPNESS_FILTER_ON) or (sharp >= MIN_SHARPNESS_VAR)

        presence_frac, motion_frac = compute_presence_motion(ref_gray, prev_gray, gray)
        vehicle_present = presence_frac >= PRESENCE_FRAC_THRESHOLD
        moving = motion_frac >= MOTION_FRAC_THRESHOLD

        if vehicle_present:
            last_presence_t = ts
        if moving:
            last_motion_t = ts

        # prebuffer always maintained
        pre_buffer.append((ts, roi.copy(), gray.copy(), is_sharp))
        while len(pre_buffer) > pre_buffer_max:
            pre_buffer.popleft()

        # immediate start when either presence or motion appears
        if (not capturing) and (vehicle_present or moving):
            capturing = True
            capture_started_t = ts
            stop_reason = ""
            pano = None
            prev_gray = None
            prev_roi = None
            residual_dx = 0.0
            smoothed_dx = 0.0
            frames_used = 0
            strips_added = 0
            strip_sum = 0

            # include prebuffer frames
            buffered = list(pre_buffer)
            for _, broi, bgray, bsharp in buffered:
                if not bsharp:
                    continue
                if prev_gray is None:
                    prev_gray = bgray
                    prev_roi = broi
                    continue
                dx, resp = estimate_dx_phase(prev_gray, bgray)
                smoothed_dx = 0.75 * smoothed_dx + 0.25 * dx
                dx_use = smoothed_dx
                residual_dx += dx_use
                step = int(clamp(abs(residual_dx), MIN_STRIP_WIDTH, MAX_STRIP_WIDTH))
                if abs(residual_dx) < MIN_STRIP_WIDTH:
                    prev_gray = bgray
                    prev_roi = broi
                    continue

                direction = DIRECTION if DIRECTION in (-1, 1) else (1 if dx_use >= 0 else -1)
                center_x = w // 2
                if dx_use >= 0:
                    x2 = clamp(center_x, 1, w)
                    x1 = clamp(x2 - step, 0, w - 1)
                else:
                    x1 = clamp(center_x, 0, w - 1)
                    x2 = clamp(x1 + step, x1 + 1, w)

                strip = broi[:, int(x1):int(x2)].copy()
                pano = append_strip(pano, strip, direction, SEAM_BLEND_PX)
                strips_added += 1
                frames_used += 1
                strip_sum += strip.shape[1]
                residual_dx -= np.sign(residual_dx) * step
                prev_gray = bgray
                prev_roi = broi

        if capturing:
            frames_processed += 1
            if prev_gray is None:
                prev_gray = gray
                prev_roi = roi
            else:
                if is_sharp:
                    dx, response = estimate_dx_phase(prev_gray, gray)
                    smoothed_dx = 0.75 * smoothed_dx + 0.25 * dx
                    dx_use = smoothed_dx
                    residual_dx += dx_use

                    # while-loop ensures no long zero-strip period when moving
                    while abs(residual_dx) >= MIN_STRIP_WIDTH:
                        step = int(clamp(abs(residual_dx), MIN_STRIP_WIDTH, MAX_STRIP_WIDTH))
                        direction = DIRECTION if DIRECTION in (-1, 1) else (1 if dx_use >= 0 else -1)
                        center_x = w // 2
                        if dx_use >= 0:
                            x2 = clamp(center_x, 1, w)
                            x1 = clamp(x2 - step, 0, w - 1)
                        else:
                            x1 = clamp(center_x, 0, w - 1)
                            x2 = clamp(x1 + step, x1 + 1, w)

                        strip = roi[:, int(x1):int(x2)].copy()
                        pano = append_strip(pano, strip, direction, SEAM_BLEND_PX)
                        strips_added += 1
                        frames_used += 1
                        strip_sum += strip.shape[1]
                        residual_dx -= np.sign(residual_dx) * step

                prev_gray = gray
                prev_roi = roi

            # stop only when both absent continuously for 2 sec
            presence_age = 999 if last_presence_t is None else (ts - last_presence_t)
            motion_age = 999 if last_motion_t is None else (ts - last_motion_t)
            if presence_age >= NO_MOTION_STOP_SEC and motion_age >= NO_MOTION_STOP_SEC:
                capturing = False
                stop_reason = "presence_and_motion_absent"

                if pano is not None and pano.size > 0:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(OUT_DIR, f"truck_profile_{stamp}.png")
                    cv2.imwrite(out_path, pano)

                    dur = 0.0 if capture_started_t is None else (ts - capture_started_t)
                    avg_w = 0.0 if strips_added == 0 else (strip_sum / float(strips_added))
                    print(
                        f"[SAVE] {out_path} | frames_received={grabber.total_received} frames_processed={frames_processed} "
                        f"frames_used={frames_used} strips_added={strips_added} avg_strip_w={avg_w:.2f} "
                        f"final_size={pano.shape[1]}x{pano.shape[0]} stop_reason={stop_reason} duration={dur:.2f}s"
                    )

                # reset session state only
                pano = None
                prev_gray = None
                prev_roi = None
                residual_dx = 0.0
                smoothed_dx = 0.0
                frames_processed = 0
                frames_used = 0
                strips_added = 0
                strip_sum = 0
                capture_started_t = None
                pre_buffer.clear()

        # preview
        if DEBUG_DISPLAY:
            disp = roi.copy()
            presence_txt = "ON" if vehicle_present else "OFF"
            motion_txt = "ON" if moving else "OFF"
            cap_txt = "ON" if capturing else "OFF"
            pano_w = 0 if pano is None else pano.shape[1]
            direction_live = DIRECTION if DIRECTION in (-1, 1) else (1 if smoothed_dx >= 0 else -1)
            strip_dbg = int(clamp(abs(residual_dx), 0, MAX_STRIP_WIDTH))

            cv2.putText(disp, f"presence={presence_txt} ({presence_frac:.4f}) motion={motion_txt} ({motion_frac:.4f})",
                        (10, 30), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(disp, f"dx={smoothed_dx:.2f} strip_w={strip_dbg} dir={direction_live} capture={cap_txt}",
                        (10, 60), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, f"frames_recv={grabber.total_received} strips={strips_added} pano_w={pano_w} q={grabber.qsize()}",
                        (10, 90), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW, resize_keep(disp, PREVIEW_SCALE))

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    grabber.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
