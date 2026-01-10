import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, Counter
import time
import os
from threading import Thread

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
    "font_family": "'Segoe UI Emoji', 'Apple Color Emoji', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
}

# ==============================================================================
# --- SETTINGS & CONSTANTS ---
# ==============================================================================
DEFAULT_CAMERA_IP = "http://10.27.181.76:8080/video" 

# ==============================================================================

# --- CSS STYLES ---
st.markdown(f"""
    <style>
    /* Global App Style */
    .stApp {{
        background-color: {THEME['background_color']};
        color: {THEME['text_color']};
        font-family: {THEME['font_family']};
    }}

    /* ESTILO DE BOTONES */
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
    
    /* Estado Hover: ROJO INTENSO */
    .stButton>button:hover {{
        background-color: {THEME['button_hover']};
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        transform: translateY(-1px);
    }}
    
    /* ESTILO DE NOTIFICACIONES (TOASTS) - AMARILLO */
    div[data-testid="stToast"] {{
        background-color: #FFFBEB !important;
        color: #92400E !important;
        border: 1px solid #F59E0B !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        padding: 16px !important;
    }}
    
    /* Icono del Toast */
    div[data-testid="stToast"] svg {{
        fill: #F59E0B !important;
    }}

    /* Pattern Cards Fixed */
    .pattern-card {{ 
        border: 1px solid {THEME['card_border']}; 
        border-radius: 8px; 
        text-align: center; 
        background-color: {THEME['card_background']}; 
        font-weight: bold; 
        color: #6B7280; 
        padding: 15px 2px; 
        margin: 2px; 
        font-size: 14px; 
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; 
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }}
    
    .pattern-filled {{ 
        border: 2px solid #10B981; 
        background-color: #D1FAE5; 
        color: #065F46; 
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); 
    }}
    
    .pattern-error {{ 
        border: 2px solid #EF4444; 
        background-color: #FEE2E2; 
        color: #991B1B; 
    }}
    
    /* Notifications Static Box (Feedback Area) */
    .notify-correct {{ 
        padding: 12px; background-color: #D1FAE5; color: #065F46; 
        border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px;
        border: 1px solid #10B981;
    }}
    .notify-wrong {{ 
        padding: 12px; background-color: #FEE2E2; color: #991B1B; 
        border-radius: 6px; text-align: center; font-weight: bold; margin-bottom: 10px;
        border: 1px solid #EF4444;
    }}
    
    .success-box {{ padding: 15px; background-color: #D1FAE5; color: #065F46; border-radius: 5px; text-align: center; }}
    .fail-box {{ padding: 15px; background-color: #FEE2E2; color: #991B1B; border-radius: 5px; text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. SHAPE DETECTION LOGIC
# ==========================================

def get_shape_name(hull, approx, contour, line_ratio):
    num_vertices = len(approx)
    x, y, w, h = cv2.boundingRect(approx)
    
    # Aspect Ratio
    aspect_ratio = float(w) / h if w > h else float(h) / w
    
    # Solidity
    area_hull = cv2.contourArea(hull)
    area_cnt = cv2.contourArea(contour)
    if area_hull == 0: return "Unidentified"
    solidity = float(area_cnt) / area_hull

    # Circularity (To distinguish Circle vs Pentagon)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return "Unidentified"
    circularity = 4 * np.pi * area_cnt / (perimeter * perimeter)

    # --- FILTERS ---
    if solidity < 0.8: return "Unidentified"

    # --- CLASSIFICATION ---

    # 1. LINE DETECTION
    if aspect_ratio > line_ratio:
        return "Line"

    # 2. GEOMETRIC SHAPES
    if num_vertices == 3:
        return "Triangle"

    elif num_vertices == 4:
        if 0.85 <= aspect_ratio <= 1.15:
            return "Square"
        else:
            return "Rectangle"

    elif num_vertices == 5:
        # If it has 5 vertices but is very round, it's likely a circle approximated poorly
        if circularity > 0.82: 
            return "Circle"
        else:
            return "Pentagon"

    elif num_vertices > 5:
        # Standard Circle Check
        if 0.7 <= circularity <= 1.2: 
            return "Circle"
    
    return "Unidentified"

def process_security_frame(frame, min_area, line_ratio, epsilon_coeff, processing_size):
    # 1. Resize (Optimization)
    small_frame = cv2.resize(frame, processing_size)
    
    # 2. Preprocessing (Exact Match to Mini Script)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Threshold (19, 16 params from script)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   19, 16)
    
    # Kernel Definitions
    kernel_noise = np.ones((1, 1), np.uint8)
    kernel_dilate = np.ones((5, 5), np.uint8) # Thicker lines to connect gaps
    
    # Morphological Operations
    # Remove noise
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_noise, iterations=2)
    # Thicken lines
    dilated = cv2.dilate(opening, kernel_dilate, iterations=2)
    
    # 3. Contour Detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_shape = "Unidentified"
    max_area = 0
    annotated = small_frame.copy() 
    
    # Scale adjustment for min_area (Streamlit specific helper)
    current_pixels = processing_size[0] * processing_size[1]
    scale_factor = current_pixels / 76800 
    adjusted_min = min_area * scale_factor

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter small noise
        if area > adjusted_min:
            # Convex Hull (Rubber band wrapper) - Key for hand drawings
            hull = cv2.convexHull(cnt)
            
            # Use HULL perimeter for epsilon, not contour perimeter
            peri = cv2.arcLength(hull, True)
            epsilon = epsilon_coeff * peri 
            
            # Approximate on HULL
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Identify Shape
            name = get_shape_name(hull, approx, cnt, line_ratio)
            
            if name != "Unidentified":
                # Draw Hull in Green
                cv2.drawContours(annotated, [hull], -1, (0, 255, 0), 2)
                
                # Draw Vertices in Red
                for p in approx:
                    cv2.circle(annotated, (p[0][0], p[0][1]), 4, (0, 0, 255), -1)

                x, y, w, h = cv2.boundingRect(approx)
                cv2.putText(annotated, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                if area > max_area:
                    max_area = area
                    best_shape = name

    return annotated, best_shape

# ==========================================
# 2. PADEL TRACKER
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists('yolov8n-pose.engine'):
        return YOLO('yolov8n-pose.engine', task='pose')
    return YOLO('yolov8n-pose.pt')

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

def run_padel_tracker(source_url, placeholders):
    video_ph, photo_ph, verdict_ph = placeholders
    src = int(source_url) if str(source_url).isdigit() else source_url
    cap = cv2.VideoCapture(src)
    
    model = load_model()
    green_lower = np.array([37, 61, 100])
    green_upper = np.array([54, 138, 226])
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    tracker = KalmanTracker()
    visual_trail = deque(maxlen=20)
    prev_ball_x = None; prev_ball_y = None; prev_velocity = 0; hit_cooldown = 0
    frame_count = 0
    
    stop_btn = st.sidebar.button("⏹️ STOP TRACKING", width="stretch")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break

        frame_proc = cv2.resize(frame, (640, 360))
        frame_count += 1
        
        waist_y = None
        if frame_count % 5 == 0:
            results = model(frame_proc, device=0, conf=0.5, max_det=1, verbose=False)
            if results and results[0].keypoints:
                pts = results[0].keypoints.xy[0].cpu().numpy()
                if len(pts) > 12:
                    hips = [p[1] for p in [pts[11], pts[12]] if p[1] > 0]
                    if hips: waist_y = int(sum(hips)/len(hips)) * 2

        blurred = cv2.GaussianBlur(frame_proc, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_c = cv2.inRange(hsv, green_lower, green_upper)
        mask_m = fgbg.apply(blurred)
        mask_m = cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask_final = cv2.bitwise_and(mask_c, mask_c, mask=mask_m)
        contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bx, by = None, None
        detected = False
        px, py = tracker.predict()
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.5 < w/h < 1.5:
                    bx, by = x+w//2, y+h//2
                    tracker.correct(bx, by)
                    detected = True
                    break 
        
        dx, dy = (bx, by) if detected else (int(px), int(py))
        
        if detected:
            visual_trail.append((dx, dy))
            if prev_ball_x:
                dist = np.sqrt((dx-prev_ball_x)**2 + (dy-prev_ball_y)**2)
                accel = dist - prev_velocity
                if accel > 8 and hit_cooldown == 0:
                    hit_cooldown = 15
                    snap = frame.copy()
                    dx_hd, dy_hd = dx*2, dy*2 
                    cv2.circle(snap, (dx_hd, dy_hd), 20, (0,0,255), 3)
                    snap_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
                    photo_ph.image(snap_rgb, caption="Impact", width="stretch")
                    verdict_ph.markdown('<div class="notify-correct"><h1>VALID</h1></div>', unsafe_allow_html=True)
            
            prev_velocity = dist if prev_ball_x else 0
            prev_ball_x, prev_ball_y = dx, dy
        else:
            prev_velocity = 0
            
        if hit_cooldown > 0: hit_cooldown -= 1
        
        for i in range(1, len(visual_trail)):
            if visual_trail[i-1] and visual_trail[i]:
                cv2.line(frame_proc, visual_trail[i-1], visual_trail[i], (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame_proc, [cv2.IMWRITE_JPEG_QUALITY, 50])
        video_ph.image(buffer.tobytes(), channels="BGR", width="stretch")
        
    cap.release()

# ==========================================
# 3. STATE MANAGEMENT
# ==========================================
if 'app_state' not in st.session_state: st.session_state['app_state'] = 'SETUP_PASSWORD' 
if 'secret_password' not in st.session_state: st.session_state['secret_password'] = []
if 'input_attempt' not in st.session_state: st.session_state['input_attempt'] = []
if 'pending_capture' not in st.session_state: st.session_state['pending_capture'] = False
if 'msg_feedback' not in st.session_state: st.session_state['msg_feedback'] = None

# ==========================================
# 4. UI: SECURITY SYSTEM (SETUP + VERIFY)
# ==========================================
def show_security_system():
    # Dynamic Headers
    if st.session_state['app_state'] == 'SETUP_PASSWORD':
        st.title("🔒 Security Setup: Create Password")
        current_target = st.session_state['secret_password']
        color_class = "pattern-filled"
    else:
        st.title("🛡️ Security Check: Unlock System")
        current_target = st.session_state['input_attempt']
        color_class = "pattern-filled"

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        cam_url = st.text_input("Camera Source (IP Webcam)", DEFAULT_CAMERA_IP)
        
        st.divider()
        st.markdown("### 📡 Video Quality & Speed")
        quality_level = st.slider("Quality Level", 1, 5, 5, help="Higher Quality = Lower Speed (and vice versa). Adjust if video lags.")
        quality_map = { 1: (240, 180, 30), 2: (320, 240, 50), 3: (400, 300, 70), 4: (480, 360, 80), 5: (640, 480, 90) }
        proc_size_w, proc_size_h, jpeg_q = quality_map[quality_level]
        PROCESSING_SIZE = (proc_size_w, proc_size_h)

        st.divider()
        st.markdown("### 🔍 Detection Tuning") 
        min_area = st.slider("Min Area", 100, 4000, 1000, help="Minimum size of the shape to be detected. Increase if noise is detected.")
        line_ratio = st.slider("Line Ratio", 3.0, 10.0, 6.5, help="Threshold to distinguish a Line from a Rectangle. Higher value means the shape must be thinner/longer to be a Line.")
        epsilon_coeff = st.slider("Polygonal Precision (Epsilon)", 0.01, 0.10, 0.035, format="%.3f", help="Smoothing factor for contours. Lower values (0.01) keep more details/vertices. Higher values (0.10) simplify shapes into basic polygons.")

    # --- MAIN LAYOUT ---
    col_vid, col_ctrl = st.columns([2, 1])

    with col_ctrl:
        st.markdown("<h3 style='text-align: center;'>Control Panel</h3>", unsafe_allow_html=True)

        # 1. Pattern Visualization
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                txt = "..."
                css = "pattern-card"
                if i < len(current_target):
                    txt = current_target[i]
                    css += f" {color_class}"
                    if st.session_state['app_state'] == 'VERIFY_PASSWORD':
                         if i < len(st.session_state['secret_password']):
                             if txt != st.session_state['secret_password'][i]:
                                 css = "pattern-card pattern-error"
                st.markdown(f'<div class="{css}">{txt}</div>', unsafe_allow_html=True)
        
        st.write("") 
        
        # --- FEEDBACK ---
        if st.session_state['msg_feedback']:
            msg, mtype = st.session_state['msg_feedback']
            if mtype == "SUCCESS": st.markdown(f'<div class="notify-correct">{msg}</div>', unsafe_allow_html=True)
            elif mtype == "ERROR": st.markdown(f'<div class="notify-wrong">{msg}</div>', unsafe_allow_html=True)
            st.session_state['msg_feedback'] = None

        st.write("") 

        # --- CAPTURE BUTTON ---
        if st.button("📸 CAPTURE SHAPE", type="primary", width="stretch"):
            st.session_state['pending_capture'] = True

        st.divider()
        
        # 2. Flow Buttons 
        if st.session_state['app_state'] == 'SETUP_PASSWORD':
            if st.button("🔐 SAVE PASSWORD", width="stretch"):
                if len(st.session_state['secret_password']) == 4:
                    st.session_state['app_state'] = 'VERIFY_PASSWORD'
                    st.session_state['input_attempt'] = []
                    st.session_state['msg_feedback'] = None
                    st.toast("Password Secured & Saved!", icon="🔐")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.toast("Password incomplete! Need 4 shapes.", icon="⚠️")
            
            # --- BUTTON DELETE ---
            if len(st.session_state['secret_password']) > 0:
                if st.button("🔙 DELETE LAST SHAPE", width="stretch"):
                    st.session_state['secret_password'].pop()
                    st.toast("Shape deleted", icon="🗑️")
                    st.rerun()
                    
        elif st.session_state['app_state'] == 'VERIFY_PASSWORD':
            if st.button("🔄 RESET PASSWORD", width="stretch"):
                st.session_state['app_state'] = 'SETUP_PASSWORD'
                st.session_state['secret_password'] = []
                st.session_state['input_attempt'] = []
                st.session_state['msg_feedback'] = None
                st.rerun()

    with col_vid:
        st.markdown("<h3 style='text-align: center;'>Live Transmission</h3>", unsafe_allow_html=True)
        cam_ph = st.empty()

    # --- CAMERA LOOP ---
    src = int(cam_url) if str(cam_url).isdigit() else cam_url
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        st.error(f"❌ Error connecting to: {src}")
        return

    loop_active = True
    
    while loop_active:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Video signal lost...")
            break

        # Process
        annotated, shape_name = process_security_frame(frame, min_area, line_ratio, epsilon_coeff, PROCESSING_SIZE)

        # Logic
        if st.session_state['pending_capture']:
            detected = shape_name
            st.session_state['pending_capture'] = False 
            
            if detected == "Unidentified":
                st.toast("No valid shape detected", icon="🚫")
            else:
                # SETUP
                if st.session_state['app_state'] == 'SETUP_PASSWORD':
                    if len(st.session_state['secret_password']) < 4:
                        st.session_state['secret_password'].append(detected)
                        st.toast(f"Captured: {detected}", icon="📸")
                        st.rerun()
                    else:
                        st.toast("Maximum 4 shapes reached!", icon="⚠️") 
                
                # VERIFY
                elif st.session_state['app_state'] == 'VERIFY_PASSWORD':
                    if len(st.session_state['input_attempt']) < 4:
                        st.session_state['input_attempt'].append(detected)
                        
                        idx = len(st.session_state['input_attempt']) - 1
                        correct = st.session_state['secret_password'][idx]
                        
                        if detected == correct:
                            st.session_state['msg_feedback'] = (f"MATCH: {detected}", "SUCCESS")
                            st.toast(f"Match: {detected}", icon="✅")
                            
                            if len(st.session_state['input_attempt']) == 4:
                                st.balloons()
                                time.sleep(1)
                                st.session_state['app_state'] = 'MAIN_MENU'
                                st.session_state['msg_feedback'] = None
                                st.rerun()
                        else:
                            st.session_state['msg_feedback'] = (f"WRONG: {detected}", "ERROR")
                            st.toast(f"Wrong: {detected}", icon="❌")
                            time.sleep(0.5)
                            st.session_state['input_attempt'] = []
                            st.rerun()
                        st.rerun()

        # Display
        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        cam_ph.image(buffer.tobytes(), channels="BGR", width="stretch")
        
    cap.release()

# ==========================================
# 5. UI: MAIN MENU (GATEWAY)
# ==========================================
def show_main_menu():
    st.title("🎛️ Padel AI Dashboard")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎾 Tracker System")
        if st.button("🚀 LAUNCH TRACKER", type="primary", width="stretch"):
            st.toast("Tracker feature not available in this version", icon="🚧")
    with col2:
        st.subheader("🔒 Security")
        if st.button("🔒 LOCK SYSTEM", width="stretch"):
            st.session_state['app_state'] = 'VERIFY_PASSWORD' 
            st.session_state['input_attempt'] = []
            st.session_state['msg_feedback'] = None
            st.rerun()
            
        if st.button("🔐 RESET PASSWORD", width="stretch"):
            st.session_state['app_state'] = 'SETUP_PASSWORD'
            st.session_state['secret_password'] = []
            st.session_state['input_attempt'] = []
            st.session_state['msg_feedback'] = None
            st.rerun()


# ==========================================
# 6. ROUTER
# ==========================================
if st.session_state['app_state'] == 'PADEL_TRACKER':
    run_padel_tracker(DEFAULT_CAMERA_IP, (st.empty(), st.empty(), st.empty()))
elif st.session_state['app_state'] == 'MAIN_MENU':
    show_main_menu()
else:
    show_security_system()




