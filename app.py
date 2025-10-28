

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Global variables
current_filter = 'none'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Cache for expensive operations
filter_cache = {}
lock = threading.Lock()

class OptimizedFilters:
    @staticmethod
    def cyberpunk(frame):
        """Cyberpunk neon effect - optimized"""
        # Downscale for processing, upscale after
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        
        b, g, r = cv2.split(small)
        b = np.clip(b.astype(np.int16) + 40, 0, 255).astype(np.uint8)
        r = np.clip(r.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        small = cv2.merge([b, g, r])
        
        # Edges
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored[:,:,0] = edges
        edges_colored[:,:,1] = edges
        small = cv2.addWeighted(small, 0.85, edges_colored, 0.15, 0)
        
        return cv2.resize(small, (w, h))
    
    @staticmethod
    def oil_painting(frame):
        """Oil painting effect - optimized"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        small = cv2.bilateralFilter(small, 9, 75, 75)
        small = small // 32 * 32 + 16
        return cv2.resize(small, (w, h))
    
    @staticmethod
    def cinematic(frame):
        """Cinematic color grading - optimized"""
        frame = frame.astype(np.float32) / 255.0
        frame[:,:,0] = np.clip(frame[:,:,0] * 1.2, 0, 1)
        frame[:,:,1] = np.clip(frame[:,:,1] * 0.95, 0, 1)
        frame[:,:,2] = np.clip(frame[:,:,2] * 1.15, 0, 1)
        
        # Simplified vignette
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        vignette = 1 - np.clip((radius / (min(h, w) * 0.7) - 0.5), 0, 1) * 0.6
        frame = frame * vignette[:,:,np.newaxis]
        
        return (frame * 255).astype(np.uint8)
    
    @staticmethod
    def sketch(frame):
        """Sketch effect - optimized"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def infrared(frame):
        """Thermal effect - optimized"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.convertScaleAbs(colored, alpha=1.2, beta=10)
    
    @staticmethod
    def neon_glow(frame):
        """Neon glow - optimized"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        hsv[:,:,2] = np.clip(hsv[:,:,2].astype(np.int16) + 30, 0, 255).astype(np.uint8)
        small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        glow = cv2.GaussianBlur(small, (15, 15), 0)
        small = cv2.addWeighted(small, 0.6, glow, 0.4, 0)
        
        return cv2.resize(small, (w, h))
    
    @staticmethod
    def matrix(frame):
        """Matrix effect - optimized"""
        green_tint = frame.copy()
        green_tint[:,:,0] = (green_tint[:,:,0] * 0.3).astype(np.uint8)
        green_tint[:,:,1] = np.clip(green_tint[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        green_tint[:,:,2] = (green_tint[:,:,2] * 0.3).astype(np.uint8)
        return green_tint
    
    @staticmethod
    def face_detection_filter(frame):
        """Face detection - optimized"""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//2, h//2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w_face, h_face) in faces:
            # Scale back to original size
            x, y, w_face, h_face = x*2, y*2, w_face*2, h_face*2
            cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 255), 3)
            cv2.putText(frame, 'DETECTED', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def vaporwave(frame):
        """Vaporwave effect - optimized"""
        frame = frame.astype(np.float32)
        frame[:,:,0] = np.clip(frame[:,:,0] * 1.3, 0, 255)
        frame[:,:,1] = frame[:,:,1] * 0.9
        frame[:,:,2] = np.clip(frame[:,:,2] * 1.2, 0, 255)
        return frame.astype(np.uint8)
    
    @staticmethod
    def hdr_effect(frame):
        """HDR effect - optimized"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def vintage_film(frame):
        """Vintage film - optimized"""
        kernel = np.array([[0.272, 0.534, 0.131],
                         [0.349, 0.686, 0.168],
                         [0.393, 0.769, 0.189]])
        sepia = cv2.transform(frame, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        # Simplified vignette
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        vignette = 1 - np.clip((radius / (min(h, w) * 0.7) - 0.3), 0, 1) * 0.7
        sepia = (sepia * vignette[:,:,np.newaxis]).astype(np.uint8)
        
        return sepia

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # Optimized resolution
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        
        filters = OptimizedFilters()
        
        # Apply selected filter
        with lock:
            filter_name = current_filter
        
        if filter_name == 'cyberpunk':
            frame = filters.cyberpunk(frame)
        elif filter_name == 'oil_painting':
            frame = filters.oil_painting(frame)
        elif filter_name == 'cinematic':
            frame = filters.cinematic(frame)
        elif filter_name == 'sketch':
            frame = filters.sketch(frame)
        elif filter_name == 'infrared':
            frame = filters.infrared(frame)
        elif filter_name == 'neon_glow':
            frame = filters.neon_glow(frame)
        elif filter_name == 'matrix':
            frame = filters.matrix(frame)
        elif filter_name == 'face_detection':
            frame = filters.face_detection_filter(frame)
        elif filter_name == 'vaporwave':
            frame = filters.vaporwave(frame)
        elif filter_name == 'hdr':
            frame = filters.hdr_effect(frame)
        elif filter_name == 'vintage_film':
            frame = filters.vintage_film(frame)
        
        # Optimized JPEG encoding
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_filter/<filter_name>')
def set_filter(filter_name):
    global current_filter
    with lock:
        current_filter = filter_name
    return jsonify({'status': 'success', 'filter': filter_name})

@app.route('/get_filter')
def get_filter():
    with lock:
        return jsonify({'filter': current_filter})

@app.route('/filters')
def get_filters():
    filters = [
        {'id': 'none', 'name': 'Original', 'category': 'basic'},
        {'id': 'face_detection', 'name': 'Face Detect', 'category': 'ai'},
        {'id': 'cyberpunk', 'name': 'Cyberpunk', 'category': 'artistic'},
        {'id': 'cinematic', 'name': 'Cinematic', 'category': 'professional'},
        {'id': 'oil_painting', 'name': 'Oil Paint', 'category': 'artistic'},
        {'id': 'sketch', 'name': 'Sketch', 'category': 'artistic'},
        {'id': 'neon_glow', 'name': 'Neon Glow', 'category': 'effects'},
        {'id': 'vaporwave', 'name': 'Vaporwave', 'category': 'aesthetic'},
        {'id': 'matrix', 'name': 'Matrix', 'category': 'effects'},
        {'id': 'infrared', 'name': 'Thermal', 'category': 'professional'},
        {'id': 'hdr', 'name': 'HDR Pro', 'category': 'professional'},
        {'id': 'vintage_film', 'name': 'Vintage', 'category': 'aesthetic'},
    ]
    return jsonify(filters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)