import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import os
import sys
from threading import Thread
import torch

# ==============================================================================
# --- 1. CONFIGURACIÓN DE PÁGINA Y TEMA VISUAL ---
# ==============================================================================
st.set_page_config(page_title="Padel AI System", page_icon="🎾", layout="wide")

THEME = {
    "background_color": "#FFFFFF",      
    "text_color": "#1F2937",            
    "card_background": "#F3F4F6",       
    "card_border": "#D1D5DB",           
    "button_primary": "#2563EB",         
    "button_hover": "#EF4444",            
    "button_text": "#FFFFFF",            
    "font_family": "'Segoe UI Emoji', 'Apple Color Emoji', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
}

st.markdown(f"""
    <style>
    .stApp {{ background-color: {THEME['background_color']}; color: {THEME['text_color']}; font-family: {THEME['font_family']}; }}
    .stButton>button {{ width: 100%; border-radius: 6px; height: 3em; font-weight: 600; font-size: 16px; background-color: {THEME['button_primary']}; color: {THEME['button_text']}; border: none; transition: all 0.3s ease; }}
    .stButton>button:hover {{ background-color: {THEME['button_hover']}; color: #FFFFFF; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); transform: translateY(-1px); }}
    div[data-testid="stToast"] {{ background-color: #FFFBEB !important; color: #92400E !important; border: 1px solid #F59E0B !important; border-radius: 8px !important; font-weight: 600 !important; padding: 16px !important; }}
    div[data-testid="stToast"] svg {{ fill: #F59E0B !important; }}
    .pattern-card {{ border: 1px solid {THEME['card_border']}; border-radius: 8px; text-align: center; background-color: {THEME['card_background']}; font-weight: bold; color: #6B7280; padding: 15px 2px; margin: 2px; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }}
    .pattern-filled {{ border: 2px solid #10B981; background-color: #D1FAE5; color: #065F46; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); }}
    .pattern-error {{ border: 2px solid #EF4444; background-color: #FEE2E2; color: #991B1B; }}
    .notify-correct {{ padding: 12px; background-color: #D1FAE5; color: #065F46; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #10B981; }}
    .notify-wrong {{ padding: 12px; background-color: #FEE2E2; color: #991B1B; border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid #EF4444; }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# --- 2. CLASES CORE ---
# ==============================================================================

class CameraStream:
    def __init__(self, src=0):
        self.src = src
        if os.name == 'nt' and (isinstance(src, int) or str(src).isdigit()):
            self.stream = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        else:
            self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.ret, self.frame) = self.stream.read()
        self.stopped = False
        self.started = False

    def start(self):
        if self.started: return self
        self.started = True
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (ret, frame) = self.stream.read()
            if ret:
                self.frame = frame
                self.ret = ret
                time.sleep(0.005) 
            else:
                self.ret = False
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

@st.cache_resource
def get_shared_camera(source_index):
    cam = CameraStream(source_index)
    cam.start()
    return cam

class KalmanTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.00003
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1
    def predict(self):
        prediction = self.kalman.predict()
        return prediction[0], prediction[1]
    def correct(self, x, y):
        self.kalman.correct(np.array([[np.float32(x)], [np.float32(y)]]))

@st.cache_resource
def load_model():
    if os.path.exists('yolov8n-pose.engine'):
        return YOLO('yolov8n-pose.engine', task='pose')
    return YOLO('yolov8n-pose.pt')

# ==============================================================================
# --- 3. LÓGICA DE SEGURIDAD ---
# ==============================================================================

def get_shape_name(hull, approx, contour, line_ratio):
    num_vertices = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h if w > h else float(h) / w
    area_hull = cv2.contourArea(hull)
    area_cnt = cv2.contourArea(contour)
    if area_hull == 0: return "Unidentified"
    solidity = float(area_cnt) / area_hull
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return "Unidentified"
    circularity = 4 * np.pi * area_cnt / (perimeter * perimeter)

    if solidity < 0.8: return "Unidentified"
    if aspect_ratio > line_ratio: return "Line"
    if num_vertices == 3: return "Triangle"
    elif num_vertices == 4: return "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
    elif num_vertices == 5: return "Circle" if circularity > 0.82 else "Pentagon"
    elif num_vertices > 5:
        if 0.7 <= circularity <= 1.2: return "Circle"
    return "Unidentified"

def process_security_frame(frame, min_area, line_ratio, epsilon_coeff, processing_size):
    if frame is None or frame.size == 0: return None, "Unidentified"
    small_frame = cv2.resize(frame, processing_size)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 16)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations=2)
    dilated = cv2.dilate(opening, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_shape = "Unidentified"; max_area = 0; annotated = small_frame.copy() 
    current_pixels = processing_size[0] * processing_size[1]
    scale_factor = current_pixels / 76800 
    adjusted_min = min_area * scale_factor

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > adjusted_min:
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            epsilon = epsilon_coeff * peri 
            approx = cv2.approxPolyDP(hull, epsilon, True)
            name = get_shape_name(hull, approx, cnt, line_ratio)
            if name != "Unidentified":
                cv2.drawContours(annotated, [hull], -1, (0, 255, 0), 2)
                for p in approx: cv2.circle(annotated, (p[0][0], p[0][1]), 4, (0, 0, 255), -1)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(annotated, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if area > max_area: max_area = area; best_shape = name
    return annotated, best_shape

# ==============================================================================
# --- 4. LÓGICA DE PADEL ---
# ==============================================================================

def get_player_zone(results, shape_img): 
    h, w = shape_img[:2]
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    waist_y = None; ground_y = None
    if not results or len(results) == 0: return zone_mask, None, None, None
    if not hasattr(results[0], 'keypoints') or results[0].keypoints is None: return zone_mask, None, None, None
    if results[0].keypoints.xy.numel() == 0: return zone_mask, None, None, None
    points = results[0].keypoints.xy[0].cpu().numpy() 
    if len(points) < 17: return zone_mask, None, None, None
    valid_points = points[np.all(points > 0, axis=1)]
    if len(valid_points) > 0:
        hips = []
        if points[11][1] > 0: hips.append(points[11][1])
        if points[12][1] > 0: hips.append(points[12][1])
        if len(hips) > 0: waist_y = int(sum(hips) / len(hips))
        feet = []
        if points[15][1] > 0: feet.append(points[15][1])
        if points[16][1] > 0: feet.append(points[16][1])
        if len(feet) > 0: ground_y = int(max(feet) - 20) 
        elif waist_y: ground_y = int(waist_y * 1.6)
        min_y, max_y = np.min(valid_points[:, 1]), np.max(valid_points[:, 1])
        min_x, max_x = np.min(valid_points[:, 0]), np.max(valid_points[:, 0])
        cx = int((min_x + max_x) / 2); cy = int((min_y + max_y) / 2)
        radius = int((max_y - min_y) * 1.2) if (max_y - min_y) > 0 else 100
        cv2.circle(zone_mask, (cx, cy), radius, 255, -1)
        return zone_mask, (cx, cy, radius), waist_y, ground_y
    return zone_mask, None, None, None

def run_padel_analysis(source, placeholders, quality_settings):
    video_ph, foto_ph, verdict_ph = placeholders
    display_w, display_h, jpeg_quality = quality_settings
    DISPLAY_SIZE = (display_w, display_h) 

    # --- RECUPERAR COLORES DE SESSION STATE ---
    # Si no existen (no ha entrado a calibrar), usar default
    if 'padel_colors' not in st.session_state:
        st.session_state['padel_colors'] = ((37, 61, 100), (54, 138, 226))
    
    (h_min, s_min, v_min), (h_max, s_max, v_max) = st.session_state['padel_colors']
    green_lower = np.array([h_min, s_min, v_min])
    green_upper = np.array([h_max, s_max, v_max])

    if 'global_camera' in st.session_state and st.session_state['global_camera'] is not None:
        if st.session_state['global_camera'].src != source:
            st.session_state['global_camera'].stop()
            st.session_state['global_camera'] = None
            time.sleep(0.5)

    if 'global_camera' not in st.session_state or st.session_state['global_camera'] is None:
        st.session_state['global_camera'] = CameraStream(source).start()
    
    cap_obj = st.session_state['global_camera']
    model = load_model()
    
    # === CONFIGURACIÓN ===
    TARGET_WIDTH = 1280 
    SKIP_YOLO_FRAMES = 3   
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    tracker = KalmanTracker()
    visual_trail = deque(maxlen=20)
    history_buffer = deque(maxlen=15) 
    
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = None; old_gray = None 
    
    prev_ball_x = None; prev_ball_y = None; prev_velocity = 0; hit_cooldown = 0
    HIT_THRESHOLD = 8; HORIZONTAL_THRESHOLD = 12; FLOW_THRESHOLD = 5 
    
    frames_without_detection = 0 
    bounce_state = 0; min_y_registered = 0; frames_verifying = 0; bounce_text_timer = 0
    frame_count = 0; cached_waist_y = None; cached_ground_y = None; cached_zone_mask = None; player_detected = False
    save_counter = 0
    
    if not os.path.exists('serve_evidence'): os.makedirs('serve_evidence')

    ret = False
    for _ in range(50):
        ret, frame_raw = cap_obj.read()
        if ret: break
        time.sleep(0.1)

    if not ret:
        st.error(f"Error iniciando cámara (Fuente: {source}).")
        return

    h_raw, w_raw = frame_raw.shape[:2]
    new_h = int(TARGET_WIDTH / (w_raw / h_raw))
    first_frame_resized = cv2.resize(frame_raw, (TARGET_WIDTH, new_h))
    old_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)

    prev_frame_time = 0
    
    stop_button = st.sidebar.button("⏹️ DETENER ANÁLISIS")
    
    while not stop_button:
        ret, frame_raw = cap_obj.read()
        if not ret: 
            time.sleep(0.01)
            continue

        h_raw, w_raw = frame_raw.shape[:2]
        new_h = int(TARGET_WIDTH / (w_raw / h_raw))
        frame = cv2.resize(frame_raw, (TARGET_WIDTH, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_count += 1
        h_curr, w_curr = frame.shape[:2]
        display_frame = frame.copy()

        # 1. PLAYER
        if frame_count % SKIP_YOLO_FRAMES == 0:
            try:
                results = model(frame, device=0, conf=0.5, max_det=1, verbose=False)
                cached_zone_mask, _, cached_waist_y, cached_ground_y = get_player_zone(results, frame.shape)
                player_detected = (cached_ground_y is not None)
            except Exception: pass
        
        if cached_zone_mask is None or cached_zone_mask.shape[:2] != (h_curr, w_curr):
            cached_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        waist_y = cached_waist_y; ground_y = cached_ground_y; zone_mask = cached_zone_mask

        # 2. BALL (Con máscara dinámica)
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, green_lower, green_upper)
        mask_mov = fgbg.apply(frame_blur)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_mov = cv2.morphologyEx(mask_mov, cv2.MORPH_OPEN, kernel)
        mask_mov = cv2.dilate(mask_mov, kernel, iterations=2)
        
        potential_ball = cv2.bitwise_and(mask_color, mask_color, mask=mask_mov)
        mask_final = cv2.bitwise_and(potential_ball, potential_ball, mask=zone_mask)
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        curr_ball_y = None; cx_ball = 0; max_area = 0; ball_detected_this_frame = False 
        
        raw_pred_x, raw_pred_y = tracker.predict()
        pred_x, pred_y = int(raw_pred_x), int(raw_pred_y)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 40: 
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w)/h
                if 0.5 < aspect_ratio < 1.5: 
                    if area > max_area:
                        max_area = area; curr_ball_y = y + h//2; cx_ball = x + w//2
                        ball_detected_this_frame = True
                        tracker.correct(cx_ball, curr_ball_y)

        # 3. TRACKING
        new_flow_point = None; flow_velocity = 0 
        if p0 is not None:
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None and status[0] == 1:
                new_flow_point = p1
                fx, fy = p0[0].ravel(); nx, ny = p1[0].ravel()
                flow_velocity = np.sqrt((nx - fx)**2 + (ny - fy)**2) 
            else: new_flow_point = None

        curr_x = pred_x; curr_y = pred_y 

        if ball_detected_this_frame:
            frames_without_detection = 0
            curr_x, curr_y = cx_ball, curr_ball_y
            p0 = np.array([[[cx_ball, curr_ball_y]]], dtype=np.float32)
            visual_trail.append((cx_ball, curr_ball_y))
        elif new_flow_point is not None and frames_without_detection < 10:
            frames_without_detection += 1
            flow_x, flow_y = new_flow_point[0].ravel()
            p0 = new_flow_point.reshape(-1, 1, 2)
            visual_trail.append((int(flow_x), int(flow_y)))
        else:
            frames_without_detection += 1
            if frames_without_detection >= 10:
                curr_ball_y = None; visual_trail.clear(); curr_x, curr_y = None, None
                prev_velocity = 0; prev_ball_x = None; p0 = None
        
        history_buffer.append((frame.copy(), curr_x, curr_y, waist_y))

        # VISUALS
        for i in range(1, len(visual_trail)):
            if visual_trail[i - 1] and visual_trail[i]:
                thickness = int(np.sqrt(20 / float(len(visual_trail) - i + 1)) * 2)
                cv2.line(display_frame, visual_trail[i - 1], visual_trail[i], (0, 255, 255), thickness)

        if curr_x is not None and curr_y is not None:
            box_radius = 15
            cv2.rectangle(display_frame, (int(curr_x - box_radius), int(curr_y - box_radius)), 
                                         (int(curr_x + box_radius), int(curr_y + box_radius)), (0, 255, 0), 2)
            cv2.putText(display_frame, "BALL", (int(curr_x - 20), int(curr_y - 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 4. HIT LOGIC
        acceleration = 0
        if 'prev_ball_x' not in locals(): prev_ball_x = None

        if curr_x is not None and prev_ball_y is not None and prev_ball_x is not None:
            dx = curr_x - prev_ball_x; dy = curr_y - prev_ball_y 
            curr_velocity = np.sqrt(dx**2 + dy**2)
            acceleration = curr_velocity - prev_velocity
            
            is_hard_hit = acceleration > HIT_THRESHOLD
            is_visual_motion = flow_velocity > FLOW_THRESHOLD

            if is_hard_hit and is_visual_motion and hit_cooldown == 0:
                if waist_y and abs(curr_y - waist_y) < 180: 
                    going_down = dy > 2; is_very_vertical = (abs(dx) < HORIZONTAL_THRESHOLD) 
                    if not (going_down or is_very_vertical):
                        hit_cooldown = 15 
                        # PHOTO FINISH
                        retro_idx = -3 
                        if len(history_buffer) >= abs(retro_idx):
                            hist_frame, hist_x, hist_y, hist_waist = history_buffer[retro_idx]
                            if hist_x and hist_y and hist_waist:
                                snapshot_frame = hist_frame.copy()
                                vertical_dist = hist_y - hist_waist
                                label = "VALID" if vertical_dist > 0 else "FAULT"
                                col = (0,255,0) if vertical_dist > 0 else (0,0,255)
                                cv2.line(snapshot_frame, (0, hist_waist), (w_curr, hist_waist), (0, 255, 255), 2)
                                cv2.circle(snapshot_frame, (hist_x, hist_y), 25, (255, 255, 0), 3)
                                cv2.line(snapshot_frame, (hist_x, hist_y), (hist_x, hist_waist), col, 2)
                                cv2.putText(snapshot_frame, f"{label}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
                                save_counter += 1
                                cv2.imwrite(f"serve_evidence/serve_{save_counter}.jpg", snapshot_frame)
                                snapshot_rgb = cv2.cvtColor(snapshot_frame, cv2.COLOR_BGR2RGB)
                                foto_ph.image(snapshot_rgb, caption=f"Impact #{save_counter}", use_container_width=True)
                                if label == "VALID": verdict_ph.markdown('<div class="notify-correct"><h1>VALID</h1></div>', unsafe_allow_html=True)
                                else: verdict_ph.markdown('<div class="notify-wrong"><h1>FAULT</h1></div>', unsafe_allow_html=True)

            prev_velocity = curr_velocity; prev_ball_x = curr_x 
        else:
            prev_velocity = 0; prev_ball_x = curr_x 

        if hit_cooldown > 0: hit_cooldown -= 1
        if curr_y: prev_ball_y = curr_y

        # 5. BOUNCE
        if curr_y is not None and ground_y is not None and player_detected:
            if bounce_state == 0:
                if prev_ball_y and prev_ball_y < ground_y and curr_y >= ground_y:
                    bounce_state = 1; min_y_registered = curr_y; frames_verifying = 0
            elif bounce_state == 1:
                frames_verifying += 1
                if curr_y > min_y_registered: min_y_registered = curr_y
                if curr_y < (min_y_registered - 15): 
                    bounce_text_timer = 30; bounce_state = 0 
                if frames_verifying > 20: bounce_state = 0

        if player_detected:
            if waist_y: cv2.line(display_frame, (0, waist_y), (w_curr, waist_y), (0, 255, 255), 2)
            if ground_y: cv2.line(display_frame, (0, ground_y), (w_curr, ground_y), (0, 255, 0), 2)
        
        if bounce_text_timer > 0:
             cv2.putText(display_frame, "BOUNCE", (int(w_curr/2)-100, int(h_curr/2)), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
             bounce_text_timer -= 1

      
        if 'prev_frame_time' not in locals(): prev_frame_time = 0
        new_frame_time = time.time()
        fps_val = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
       
        cv2.rectangle(display_frame, (w_curr - 160, 10), (w_curr - 10, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, f"FPS: {int(fps_val)}", (w_curr - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        old_gray = frame_gray.copy()
        
        # --- LIMPIEZA VISUAL: SOLO VIDEO (SIN MÁSCARA) ---
        frame_vis = cv2.resize(display_frame, DISPLAY_SIZE) 
        ret, buffer = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]) 
        video_ph.image(buffer.tobytes(), channels="BGR", width="stretch")

        time.sleep(0.01)

# ==============================================================================
# --- 5. GESTIÓN DE ESTADO Y UI ---
# ==============================================================================

if 'app_state' not in st.session_state: st.session_state['app_state'] = 'SETUP_PASSWORD' 
if 'secret_password' not in st.session_state: st.session_state['secret_password'] = []
if 'input_attempt' not in st.session_state: st.session_state['input_attempt'] = []
if 'pending_capture' not in st.session_state: st.session_state['pending_capture'] = False
if 'msg_feedback' not in st.session_state: st.session_state['msg_feedback'] = None
if 'global_camera' not in st.session_state: st.session_state['global_camera'] = None
# NUEVO: Estado de colores por defecto
if 'padel_colors' not in st.session_state: st.session_state['padel_colors'] = ((37, 61, 100), (54, 138, 226))

# --- PANTALLA DE SEGURIDAD ---
def show_security_system():
    DEFAULT_SOURCE = "2"
    if st.session_state['app_state'] == 'SETUP_PASSWORD':
        st.title("🔒 Security Setup: Create Password")
        current_target = st.session_state['secret_password']
    else:
        st.title("🛡️ Security Check: Unlock System")
        current_target = st.session_state['input_attempt']

    with st.sidebar:
        st.header("⚙️ Configuration")
        cam_input = st.text_input("Camera Source", DEFAULT_SOURCE)
        st.divider()
        quality_level = st.slider("Quality Level", 1, 5, 3)
        quality_map = { 1: (240, 180, 30), 2: (320, 240, 50), 3: (400, 300, 70), 4: (480, 360, 80), 5: (640, 480, 90) }
        PROCESSING_SIZE = quality_map[quality_level][:2]
        JPEG_Q = quality_map[quality_level][2]
        st.divider()
        st.markdown("### 🔍 Detection Tuning") 
        min_area = st.slider("Min Area", 100, 4000, 1000)
        line_ratio = st.slider("Line Ratio", 3.0, 10.0, 6.5)
        epsilon_coeff = st.slider("Polygonal Precision", 0.01, 0.10, 0.035, format="%.3f")

    col_vid, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.markdown("<h3 style='text-align: center;'>Control Panel</h3>", unsafe_allow_html=True)
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                txt = "..."; css = "pattern-card"
                if i < len(current_target):
                    txt = current_target[i]; css += " pattern-filled"
                    if st.session_state['app_state'] == 'VERIFY_PASSWORD' and i < len(st.session_state['secret_password']):
                             if txt != st.session_state['secret_password'][i]: css = "pattern-card pattern-error"
                st.markdown(f'<div class="{css}">{txt}</div>', unsafe_allow_html=True)
        st.write("") 
        if st.session_state['msg_feedback']:
            msg, mtype = st.session_state['msg_feedback']
            css_class = "notify-correct" if mtype == "SUCCESS" else "notify-wrong"
            st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
            st.session_state['msg_feedback'] = None
        st.write("") 
        if st.button("📸 CAPTURE SHAPE", type="primary"):
            st.session_state['pending_capture'] = True
        st.divider()
        if st.session_state['app_state'] == 'SETUP_PASSWORD':
            if st.button("🔐 SAVE PASSWORD"):
                if len(st.session_state['secret_password']) == 4:
                    st.session_state['app_state'] = 'VERIFY_PASSWORD'
                    st.session_state['input_attempt'] = []
                    st.toast("Password Saved!", icon="🔐")
                    time.sleep(0.5); st.rerun()
                else: st.toast("Need 4 shapes.", icon="⚠️")
            if len(st.session_state['secret_password']) > 0:
                if st.button("🔙 DELETE LAST"):
                    st.session_state['secret_password'].pop(); st.rerun()
        elif st.session_state['app_state'] == 'VERIFY_PASSWORD':
            if st.button("🔄 RESET"):
                st.session_state['app_state'] = 'SETUP_PASSWORD'
                st.session_state['secret_password'] = []; st.session_state['input_attempt'] = []
                st.rerun()

    with col_vid:
        cam_ph = st.empty()

    final_source = 0
    if cam_input.isdigit(): final_source = int(cam_input)
    elif "http" in cam_input: final_source = cam_input
    else: final_source = 0 
    
    if st.session_state['global_camera'] is None:
        st.session_state['global_camera'] = get_shared_camera(final_source)
    elif st.session_state['global_camera'].src != final_source:
        st.session_state['global_camera'].stop()
        st.session_state['global_camera'] = get_shared_camera(final_source)
    
    cam = st.session_state['global_camera']

    prev_frame_time = 0
    
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        annotated, shape_name = process_security_frame(frame, min_area, line_ratio, epsilon_coeff, PROCESSING_SIZE)
        
        # --- BLOQUE FPS (Insertar aquí) ---
        new_frame_time = time.time()
        fps_val = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        # Si annotated existe, dibujamos sobre él
        if annotated is not None:
            h_ann, w_ann = annotated.shape[:2]
            # Ajustamos tamaño del texto según resolución pequeña de seguridad
            cv2.rectangle(annotated, (w_ann - 100, 5), (w_ann - 5, 35), (0, 0, 0), -1)
            cv2.putText(annotated, f"FPS: {int(fps_val)}", (w_ann - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # ----------------------------------

        if annotated is not None and annotated.size > 0:
            ret_enc, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ret_enc:
                cam_ph.image(buffer.tobytes(), channels="BGR", width="stretch")

        if st.session_state['pending_capture']:
            st.session_state['pending_capture'] = False 
            if shape_name == "Unidentified": st.toast("No valid shape", icon="🚫")
            else:
                if st.session_state['app_state'] == 'SETUP_PASSWORD':
                    if len(st.session_state['secret_password']) < 4:
                        st.session_state['secret_password'].append(shape_name)
                        st.toast(f"Captured: {shape_name}", icon="📸"); st.rerun()
                    else: st.toast("Maximum 4 shapes reached!", icon="⚠️") 
                elif st.session_state['app_state'] == 'VERIFY_PASSWORD':
                    if len(st.session_state['input_attempt']) < 4:
                        st.session_state['input_attempt'].append(shape_name)
                        idx = len(st.session_state['input_attempt']) - 1
                        correct = st.session_state['secret_password'][idx]
                        if shape_name == correct:
                            st.session_state['msg_feedback'] = (f"MATCH: {shape_name}", "SUCCESS")
                            if len(st.session_state['input_attempt']) == 4:
                                st.balloons(); time.sleep(1)
                                st.session_state['app_state'] = 'MAIN_MENU'
                                st.rerun()
                            else: st.rerun()
                        else:
                            st.session_state['msg_feedback'] = (f"WRONG: {shape_name}", "ERROR")
                            time.sleep(0.5); st.session_state['input_attempt'] = []; st.rerun()
        time.sleep(0.03)

# --- PANTALLA PRINCIPAL ---
def show_main_menu():
    st.title("🎛️ Padel AI Dashboard")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎾 AI Chair Umpire")
        if st.button("🚀 LAUNCH SYSTEM", type="primary"):
            st.session_state['app_state'] = 'PADEL_APP'
            st.rerun()
    with col2:
        st.subheader("🔒 Security")
        if st.button("🔒 LOCK SYSTEM"):
            st.session_state['app_state'] = 'VERIFY_PASSWORD' 
            st.session_state['input_attempt'] = []
            st.rerun()
        if st.button("🔐 RESET PASSWORD"):
            st.session_state['app_state'] = 'SETUP_PASSWORD'
            st.session_state['secret_password'] = []
            st.rerun()

# --- PANTALLA DE CALIBRACIÓN DE COLOR (CORREGIDA) ---
def show_calibration_screen():
    st.title("🎨 Calibración de Color (Pelota)")
    
    # Sliders en Sidebar
    with st.sidebar:
        st.header("Ajustes HSV")
        st.info("Mueve los sliders hasta que la pelota se vea BLANCA y el resto NEGRO.")
        
        # --- AQUÍ BORRÉ LA LÍNEA QUE DABA ERROR ---
        
        # Sliders (Cogen el valor por defecto directamente de la memoria)
        h_min = st.slider("H Min", 0, 179, st.session_state['padel_colors'][0][0])
        h_max = st.slider("H Max", 0, 179, st.session_state['padel_colors'][1][0])
        
        s_min = st.slider("S Min", 0, 255, st.session_state['padel_colors'][0][1])
        s_max = st.slider("S Max", 0, 255, st.session_state['padel_colors'][1][1])
        
        v_min = st.slider("V Min", 0, 255, st.session_state['padel_colors'][0][2])
        v_max = st.slider("V Max", 0, 255, st.session_state['padel_colors'][1][2])
        
        if st.button("💾 GUARDAR Y VOLVER", type="primary"):
            # Guardar en memoria
            st.session_state['padel_colors'] = ((h_min, s_min, v_min), (h_max, s_max, v_max))
            st.session_state['app_state'] = 'PADEL_APP'
            st.rerun()

    col_orig, col_mask = st.columns(2)
    with col_orig: st.markdown("**📷 Cámara Original**"); ph_orig = st.empty()
    with col_mask: st.markdown("**🎭 Máscara (Lo que ve la IA)**"); ph_mask = st.empty()

    # Usar cámara compartida
    if st.session_state['global_camera'] is None:
        st.session_state['global_camera'] = get_shared_camera(2) # Default 2
    cam = st.session_state['global_camera']

    while True:
        ret, frame = cam.read()
        if not ret: time.sleep(0.01); continue
        
        # Resize para velocidad
        frame_small = cv2.resize(frame, (640, 360))
        
        # Procesar
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Mostrar
        ph_orig.image(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        ph_mask.image(mask)
        
        time.sleep(0.03)

# --- PANTALLA APP PADEL (MODIFICADA) ---
def show_padel_app():
    st.title("🎾 AI Chair Umpire")
    with st.sidebar:
        st.header("Configuration")
        opcion_fuente = st.radio("Video Source", ["USB Camera", "Video File"])
        source = 2
        
        if opcion_fuente == "USB Camera":
            url_input = st.text_input("Camera Index", "2")
            if url_input.isdigit(): source = int(url_input)
            elif "http" in url_input: source = url_input
            else: source = 0
        else:
            video_path = st.text_input("File Path", "videos/video.mp4")
            source = video_path
            
        st.divider()
        
        # BOTÓN PARA IR A CALIBRAR
        if st.button("🎨 CALIBRAR COLORES (MÁSCARA)"):
            st.session_state['app_state'] = 'CALIBRATION'
            st.rerun()

        st.divider()
        quality_level = st.slider("Streaming Quality", 1, 5, 3)
        quality_map = { 1: (320, 180, 40), 2: (480, 270, 60), 3: (640, 360, 70), 4: (854, 480, 80), 5: (1280, 720, 90) }
        current_quality = quality_map[quality_level]
        st.divider()
        if st.button("⬅️ Back to Dashboard"):
            st.session_state['app_state'] = 'MAIN_MENU'
            st.rerun()

    col_video, col_stats = st.columns([3, 1], gap="medium")
    with col_video: video_placeholder = st.empty()
    with col_stats:
        st.markdown("### 📸 Last Serve")
        foto_placeholder = st.empty(); veredicto_placeholder = st.empty()

    if st.sidebar.button("▶️ START MATCH", type="primary"):
        phs = (video_placeholder, foto_placeholder, veredicto_placeholder)
        run_padel_analysis(source, phs, current_quality)

# --- ROUTER ---
if st.session_state['app_state'] == 'PADEL_APP':
    show_padel_app()
elif st.session_state['app_state'] == 'CALIBRATION':
    show_calibration_screen()
elif st.session_state['app_state'] == 'MAIN_MENU':
    show_main_menu()
else:
    show_security_system() 


