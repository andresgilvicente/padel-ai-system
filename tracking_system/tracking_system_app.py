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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Padel AI System", page_icon="🎾", layout="wide")

# ==============================================================================
# --- VISUAL THEME CONFIGURATION ---
# ==============================================================================
THEME = {
    "background_color": "#FFFFFF",      
    "text_color": "#1F2937",            
    "card_background": "#F3F4F6",       
    "card_border": "#D1D5DB",           
    "button_primary": "#2563EB",        
    "button_hover": "#EF4444",          
    "button_text": "#FFFFFF",           
    "font_family": "sans-serif"
}

# --- CSS STYLES ---
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {THEME['background_color']};
        color: {THEME['text_color']};
        font-family: {THEME['font_family']};
    }}
    .stButton>button {{ 
        width: 100%; 
        border-radius: 6px; 
        height: 3em; 
        font-weight: 600; 
        font-size: 16px; 
        background-color: {THEME['button_primary']}; 
        color: {THEME['button_text']};
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {THEME['button_hover']};
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        transform: translateY(-1px);
    }}
    div[data-testid="stToast"] {{
        background-color: #FFFBEB !important;
        color: #92400E !important;
        border: 1px solid #F59E0B !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 16px !important;
    }}
    .success-box {{ padding: 15px; background-color: #D1FAE5; color: #065F46; border-radius: 5px; text-align: center; border: 1px solid #10B981; font-weight: bold; }}
    .fail-box {{ padding: 15px; background-color: #FEE2E2; color: #991B1B; border-radius: 5px; text-align: center; border: 1px solid #EF4444; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. CORE CLASSES
# ==========================================

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if src == 0 or (isinstance(src, str) and src.isdigit()):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.stream.set(cv2.CAP_PROP_FPS, 60)
        (self.ret, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.ret, self.frame) = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

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
        cx = int((min_x + max_x) / 2)
        cy = int((min_y + max_y) / 2)
        radius = int((max_y - min_y) * 1.2) if (max_y - min_y) > 0 else 100
        
        cv2.circle(zone_mask, (cx, cy), radius, 255, -1)
        return zone_mask, (cx, cy, radius), waist_y, ground_y
    
    return zone_mask, None, None, None

# ==========================================
# 2. MAIN LOGIC (SENSOR FUSION + QUALITY CONTROL)
# ==========================================
def run_analysis_app(source, placeholders, quality_settings):
    video_ph, foto_ph, verdict_ph = placeholders
    
    # Display Settings
    display_w, display_h, jpeg_quality = quality_settings
    DISPLAY_SIZE = (display_w, display_h) 
    
    # Start Stream
    is_live = False
    cap_obj = None 

    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap_obj = CameraStream(source).start()
        is_live = True
        st.toast("System Running (Live Mode)")
    else:
        cap = cv2.VideoCapture(source)
        is_live = False

    model = load_model()

    # --- INTERNAL CONFIGURATION (Always HD for Physics) ---
    CALC_WIDTH = 1280 
    SKIP_YOLO_FRAMES = 3   
    
    green_lower = np.array([37, 61, 100])
    green_upper = np.array([54, 138, 226])
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    tracker = KalmanTracker()
    visual_trail = deque(maxlen=20)
    frames_without_detection = 0    
    MAX_MISSED_FRAMES = 10    
    history_buffer = deque(maxlen=15) 

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = None      
    old_gray = None 

    prev_ball_x = None; prev_ball_y = None; prev_velocity = 0
    hit_cooldown = 0; HIT_THRESHOLD = 8; 
    HORIZONTAL_THRESHOLD = 12  
    FLOW_THRESHOLD = 5 
    
    bounce_state = 0; min_y_registered = 0; frames_verifying = 0; bounce_text_timer = 0
    
    frame_count = 0
    cached_waist_y = None; cached_ground_y = None; cached_zone_mask = None
    player_detected = False
    prev_frame_time = 0

    if not os.path.exists('serve_evidence'): os.makedirs('serve_evidence')
    save_counter = 0

    # Init first frame
    if not is_live:
        ret, first_frame = cap.read()
        if ret:
            h_raw, w_raw = first_frame.shape[:2]
            new_h = int(CALC_WIDTH / (w_raw / h_raw))
            first_frame_resized = cv2.resize(first_frame, (CALC_WIDTH, new_h))
            old_gray = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2GRAY)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stop_button = st.sidebar.button("⏹️ STOP ANALYSIS")
    
    while not stop_button:
        if is_live:
            ret, frame_raw = cap_obj.read()
        else:
            ret, frame_raw = cap.read()
            if not ret: break 
        
        if frame_raw is None: continue

        # --- RESIZE FOR PHYSICS (HD) ---
        h_raw, w_raw = frame_raw.shape[:2]
        aspect_ratio = w_raw / h_raw
        new_h = int(CALC_WIDTH / aspect_ratio)
        frame = cv2.resize(frame_raw, (CALC_WIDTH, new_h))
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if old_gray is None: old_gray = frame_gray.copy() 

        frame_count += 1
        h_curr, w_curr = frame.shape[:2]

        # 1. PLAYER DETECTION
        if frame_count % SKIP_YOLO_FRAMES == 0:
            try:
                results = model(frame, device=0, conf=0.5, max_det=1, verbose=False)
                cached_zone_mask, _, cached_waist_y, cached_ground_y = get_player_zone(results, frame.shape)
                player_detected = (cached_ground_y is not None)
            except Exception:
                player_detected = False
        
        if cached_zone_mask is None or cached_zone_mask.shape[:2] != (h_curr, w_curr):
            cached_zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        waist_y = cached_waist_y
        ground_y = cached_ground_y
        zone_mask = cached_zone_mask

        # 2. BALL DETECTION
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
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 1.5: 
                    if area > max_area:
                        max_area = area; curr_ball_y = y + h//2; cx_ball = x + w//2
                        ball_detected_this_frame = True
                        tracker.correct(cx_ball, curr_ball_y)

        # 3. HYBRID TRACKING
        new_flow_point = None
        flow_velocity = 0 
        
        if p0 is not None:
            p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None and status[0] == 1:
                new_flow_point = p1
                fx, fy = p0[0].ravel()
                nx, ny = p1[0].ravel()
                flow_velocity = np.sqrt((nx - fx)**2 + (ny - fy)**2) 
            else:
                new_flow_point = None

        curr_x = pred_x; curr_y = pred_y 

        if ball_detected_this_frame:
            frames_without_detection = 0
            curr_x, curr_y = cx_ball, curr_ball_y
            p0 = np.array([[[cx_ball, curr_ball_y]]], dtype=np.float32)
            visual_trail.append((cx_ball, curr_ball_y))
        
        elif new_flow_point is not None and frames_without_detection < MAX_MISSED_FRAMES:
            frames_without_detection += 1
            flow_x, flow_y = new_flow_point[0].ravel()
            p0 = new_flow_point.reshape(-1, 1, 2)
            visual_trail.append((int(flow_x), int(flow_y)))
        else:
            frames_without_detection += 1
            if frames_without_detection >= MAX_MISSED_FRAMES:
                curr_ball_y = None; visual_trail.clear(); curr_x, curr_y = None, None
                prev_velocity = 0; prev_ball_x = None; p0 = None
        
        history_buffer.append((frame.copy(), curr_x, curr_y, waist_y))

        # 4. DRAW VISUALS
        for i in range(1, len(visual_trail)):
            if visual_trail[i - 1] and visual_trail[i]:
                thickness = int(np.sqrt(20 / float(len(visual_trail) - i + 1)) * 2)
                cv2.line(frame, visual_trail[i - 1], visual_trail[i], (0, 255, 255), thickness)

        if curr_x is not None and curr_y is not None:
            box_radius = 15
            top_left = (int(curr_x - box_radius), int(curr_y - box_radius))
            bottom_right = (int(curr_x + box_radius), int(curr_y + box_radius))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, "BALL", (int(curr_x - 20), int(curr_y - 25)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 5. HIT LOGIC (FUSION)
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
                        
                        # --- PHOTO FINISH ---
                        retro_idx = -3 
                        if len(history_buffer) >= abs(retro_idx):
                            hist_frame, hist_x, hist_y, hist_waist = history_buffer[retro_idx]
                            
                            if hist_x and hist_y and hist_waist:
                                snapshot_frame = hist_frame.copy()
                                h_snap, w_snap = snapshot_frame.shape[:2]
                                
                                cv2.line(snapshot_frame, (0, hist_waist), (w_snap, hist_waist), (0, 255, 255), 2)
                                cv2.circle(snapshot_frame, (hist_x, hist_y), 25, (255, 255, 0), 3)
                                
                                vertical_dist = hist_y - hist_waist
                                col = (0,255,0) if vertical_dist > 0 else (0,0,255)
                                # EVIDENCE LINE
                                cv2.line(snapshot_frame, (hist_x, hist_y), (hist_x, hist_waist), col, 2)
                                
                                save_counter += 1
                                cv2.imwrite(f"serve_evidence/serve_{save_counter}.jpg", snapshot_frame)
                                
                                snapshot_rgb = cv2.cvtColor(snapshot_frame, cv2.COLOR_BGR2RGB)
                                foto_ph.image(snapshot_rgb, caption=f"Impact #{save_counter}", use_container_width=True)
                                
                                if vertical_dist > 0:
                                    verdict_ph.markdown('<div class="success-box">VALID</div>', unsafe_allow_html=True)
                                else:
                                    verdict_ph.markdown('<div class="fail-box">FAULT</div>', unsafe_allow_html=True)

            prev_velocity = curr_velocity; prev_ball_x = curr_x 
        else:
            prev_velocity = 0; prev_ball_x = curr_x 

        if hit_cooldown > 0: hit_cooldown -= 1
        if curr_y: prev_ball_y = curr_y

        # 6. BOUNCE LOGIC
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

        # 7. FINAL RENDER (RESIZED FOR SPEED)
        if player_detected:
            if waist_y: cv2.line(frame, (0, waist_y), (w_curr, waist_y), (0, 255, 255), 2)
            if ground_y: cv2.line(frame, (0, ground_y), (w_curr, ground_y), (0, 255, 0), 2)
        
        if bounce_text_timer > 0:
             cv2.putText(frame, "BOUNCE", (int(w_curr/2)-100, int(h_curr/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
             bounce_text_timer -= 1

        new_frame_time = time.time()
        fps_val = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        cv2.rectangle(frame, (w_curr - 160, 10), (w_curr - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps_val)}", (w_curr - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        old_gray = frame_gray.copy()

        # Resizing for display performance
        frame_vis = cv2.resize(frame, DISPLAY_SIZE) 
        ret, buffer = cv2.imencode('.jpg', frame_vis, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]) 
        video_ph.image(buffer.tobytes(), channels="BGR", width="stretch")

    if is_live: cap_obj.stop()
    else: cap.release()

# ==========================================
# 3. SCREENS
# ==========================================

if 'estado' not in st.session_state: st.session_state['estado'] = 'SEGURIDAD'

def show_security_screen():
    st.title("🔒 Security Access")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("⚠️ Security Module Not Implemented yet.")
        st.image("https://placehold.co/640x360/262730/FFFFFF/png?text=SECURITY+MODULE+OFF", use_container_width=True)
    
    with col2:
        st.markdown("##### 🔐 Override")
        st.warning("Click below to enter directly.")
        st.divider()
        if st.button("🚀 ENTER SYSTEM", type="primary"):
            st.toast("Welcome!")
            time.sleep(0.5)
            st.session_state['estado'] = 'APP_IA' # Direct jump to App
            st.rerun()

def show_app_screen():
    st.title("🎾 AI Chair Umpire")
    
    with st.sidebar:
        st.header("Configuration")
        opcion_fuente = st.radio("Video Source", ["IP Camera (Mobile)", "Video File"])
        
        source = 0
        if opcion_fuente == "IP Camera (Mobile)":
            url = st.text_input("IP Webcam URL", "http://10.27.181.76:8080/video")
            source = url if "http" in url else 0
            if source == 0: st.caption("Using PC Webcam (0)")
        else:
            video_path = st.text_input("File Path", "videos/video.mp4")
            source = video_path

        st.divider()
        st.markdown("### 📡 Display Quality")
        # Slider controls visual quality vs speed
        quality_level = st.slider("Quality vs Speed", 1, 5, 3, help="1=Fastest (Low Res), 5=Best (HD). Physics always run at HD.")
        quality_map = { 1: (320, 180, 40), 2: (480, 270, 60), 3: (640, 360, 70), 4: (854, 480, 80), 5: (1280, 720, 90) }
        current_quality = quality_map[quality_level]

        st.divider()
        if st.button("⬅️ Logout"):
            st.session_state['estado'] = 'SEGURIDAD'
            st.rerun()

    col_video, col_stats = st.columns([3, 1], gap="medium")
    
    with col_video:
        video_placeholder = st.empty()
    
    with col_stats:
        st.markdown("### 📸 Last Serve")
        foto_placeholder = st.empty()
        veredicto_placeholder = st.empty()

    if st.sidebar.button("▶️ START ANALYSIS", type="primary"):
        phs = (video_placeholder, foto_placeholder, veredicto_placeholder)
        run_analysis_app(source, phs, current_quality)

# MAIN CONTROLLER
if st.session_state['estado'] == 'SEGURIDAD':
    show_security_screen()
elif st.session_state['estado'] == 'APP_IA':
    show_app_screen()