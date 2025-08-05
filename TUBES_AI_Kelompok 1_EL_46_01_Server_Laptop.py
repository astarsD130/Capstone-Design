import cv2
import socket
import pickle
import struct
import threading
import numpy as np
import time
from ultralytics import YOLO

class ObjectDetectionServer:
    def __init__(self, video_port=8888, result_port=8889):
        self.video_port = video_port
        self.result_port = result_port
        
        # Load YOLO model
        print("Loading YOLO model...")
        self.model = YOLO(r'..\TUBES_AI_Kelompok 1_EL_46_01_MODEL YOLOv8\best.pt')
        print("Model loaded successfully!")
        
        # Socket for receiving video
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(('0.0.0.0', self.video_port))
        self.video_socket.listen(1)
        
        # Socket for sending results (will be created later)
        self.result_socket = None
        
        self.running = True
        self.current_frame = None
        self.rpi_ip = None
        self.frame_lock = threading.Lock()
        self.detections_queue = []
        
        # Color correction mode
        self.color_mode = "auto"  # auto, bgr, rgb, none
        
    def receive_frame_data(self, conn, size):
        """Safely receive exact amount of data"""
        data = b""
        while len(data) < size:
            try:
                remaining = size - len(data)
                chunk_size = min(remaining, 65536)  # 64KB chunks
                packet = conn.recv(chunk_size)
                
                if not packet:
                    raise ConnectionError("Connection closed by client")
                    
                data += packet
                
            except socket.timeout:
                print("Socket timeout while receiving data")
                raise
            except Exception as e:
                print(f"Error receiving data: {e}")
                raise
                
        return data
    
    def fix_color_simple(self, frame):
        """Simple color correction based on mode"""
        if frame is None or frame.size == 0:
            return None
        
        try:
            if self.color_mode == "bgr":
                # Force RGB to BGR conversion
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            elif self.color_mode == "rgb":
                # Keep as RGB (don't convert)
                pass
                
            elif self.color_mode == "auto":
                # Analyze the frame to determine best color space
                b, g, r = cv2.split(frame)
                
                # Check if red channel is dominant (sign of BGR/RGB mixup)
                red_mean = np.mean(r)
                blue_mean = np.mean(b)
                
                if red_mean > blue_mean * 1.3:  # Red is much stronger than blue
                    # Likely need to swap channels
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    print(f"Auto-corrected: RGB->BGR (R:{red_mean:.0f} B:{blue_mean:.0f})")
            
            # Ensure proper data type
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            return frame
            
        except Exception as e:
            print(f"Error in color correction: {e}")
            return frame
    
    def receive_video_stream(self, conn):
        """Receive video frames from Raspberry Pi with simple color correction"""
        conn.settimeout(30.0)  # Longer timeout
        
        try:
            print("Starting video stream reception...")
            frame_count = 0
            
            while self.running:
                try:
                    # Receive frame size (4 bytes)
                    size_data = self.receive_frame_data(conn, 4)
                    frame_size = struct.unpack("!L", size_data)[0]  # Network byte order
                    
                    # Validate frame size (reasonable limits)
                    if frame_size > 5000000:  # 5MB limit
                        print(f"Frame size too large: {frame_size} bytes, skipping")
                        continue
                        
                    if frame_size == 0:
                        print("Received zero-size frame, skipping")
                        continue
                    
                    # Receive frame data
                    frame_data = self.receive_frame_data(conn, frame_size)
                    
                    # Verify data integrity
                    if len(frame_data) != frame_size:
                        print(f"Data size mismatch: expected {frame_size}, got {len(frame_data)}")
                        continue
                    
                    # Deserialize frame
                    try:
                        # Decode as JPEG image
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None and frame.size > 0:
                            # Apply simple color correction
                            corrected_frame = self.fix_color_simple(frame)
                            
                            if corrected_frame is not None:
                                with self.frame_lock:
                                    self.current_frame = corrected_frame.copy()
                                
                                frame_count += 1
                                
                                # Log every 60 frames
                                if frame_count % 60 == 0:
                                    print(f"Frame {frame_count}: {corrected_frame.shape}, Mode: {self.color_mode}")
                            else:
                                print(f"Frame {frame_count}: Color correction failed")
                        else:
                            print(f"Frame {frame_count}: Failed to decode")
                            
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        continue
                        
                except ConnectionError:
                    print("Connection lost")
                    break
                except socket.timeout:
                    print("Socket timeout - no data received")
                    continue
                except Exception as e:
                    print(f"Unexpected error in video reception: {e}")
                    break
                    
        except Exception as e:
            print(f"Critical error in video stream: {e}")
        finally:
            print("Video stream reception ended")
    
    def process_detection(self):
        """Process object detection on received frames"""
        print("Starting detection processing thread...")
        
        try:
            detection_count = 0
            last_process_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Process at ~5 FPS to avoid overload
                if current_time - last_process_time < 0.2:
                    time.sleep(0.01)
                    continue
                
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame.copy()
                    else:
                        frame_to_process = None
                
                if frame_to_process is not None:
                    try:
                        # Run YOLO detection
                        start_time = time.time()
                        results = self.model(frame_to_process, verbose=False, conf=0.5)
                        inference_time = time.time() - start_time
                        
                        # Parse results
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    # Get bounding box coordinates
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    confidence = box.conf[0].cpu().numpy()
                                    class_id = int(box.cls[0].cpu().numpy())
                                    class_name = self.model.names[class_id]
                                    
                                    detection = {
                                        'class': class_name,
                                        'confidence': float(confidence),
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                    }
                                    detections.append(detection)
                        
                        detection_count += 1
                        if detection_count % 10 == 0:  # Reduce logging
                            print(f"Detection {detection_count}: Found {len(detections)} objects in {inference_time:.3f}s")
                        
                        # Store detections for sending
                        self.detections_queue = detections
                        
                        # Display results on laptop
                        self.display_frame_with_detections(frame_to_process, detections, inference_time)
                        
                        last_process_time = current_time
                        
                    except Exception as e:
                        print(f"Detection processing error: {e}")
                        
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except Exception as e:
            print(f"Critical error in detection processing: {e}")
    
    def send_results_thread(self):
        """Send detection results back to Raspberry Pi in separate thread"""
        print("Starting result sender thread...")
        
        while self.running:
            try:
                if self.result_socket and self.detections_queue:
                    detections = self.detections_queue.copy()
                    
                    # Serialize detection results
                    data = pickle.dumps(detections)
                    size = len(data)
                    
                    # Send size then data
                    self.result_socket.sendall(struct.pack("!L", size))
                    self.result_socket.sendall(data)
                    
                    if len(detections) > 0:  # Only log when there are detections
                        print(f"Sent {len(detections)} detections to RPi")
                    
                    # Clear queue
                    self.detections_queue = []
                    
            except Exception as e:
                print(f"Error sending results: {e}")
                self.result_socket = None
                time.sleep(1)
            
            time.sleep(0.5)  # Send results every 500ms
    
    def display_frame_with_detections(self, frame, detections, inference_time=0):
        """Display frame with detection results on laptop"""
        if frame is None:
            return
            
        display_frame = frame.copy()
        
        # Add color info for debugging
        b, g, r = cv2.split(display_frame)
        color_info = f"Mode:{self.color_mode} R:{np.mean(r):.0f} G:{np.mean(g):.0f} B:{np.mean(b):.0f}"
        
        # Add performance info
        fps = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(display_frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f'Objects: {len(detections)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(display_frame, color_info, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(display_frame, (x1, y1-label_size[1]-10), 
                         (x1+label_size[0], y1), (0, 255, 0), -1)
            
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Object Detection - Laptop View', display_frame)
        
        # Handle key presses for color correction
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord('q'):
            self.running = False
        elif key == ord('s'):
            # Save screenshot
            filename = f'detection_screenshot_{int(time.time())}.jpg'
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('1'):
            # Set to auto mode
            self.color_mode = "auto"
            print("Color mode: AUTO")
        elif key == ord('2'):
            # Force BGR mode
            self.color_mode = "bgr"
            print("Color mode: BGR (RGB->BGR conversion)")
        elif key == ord('3'):
            # Force RGB mode (no conversion)
            self.color_mode = "rgb"
            print("Color mode: RGB (no conversion)")
        elif key == ord('4'):
            # No color correction
            self.color_mode = "none"
            print("Color mode: NONE")
    
    def connect_to_rpi_for_results(self, rpi_ip):
        """Connect to Raspberry Pi for sending results with delay"""
        # Wait a bit for RPi to be ready
        print("Waiting for RPi to be ready for results...")
        time.sleep(3)
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                print(f"Connecting to RPi for results (attempt {attempt+1}/{max_retries})...")
                self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.result_socket.settimeout(5.0)
                self.result_socket.connect((rpi_ip, self.result_port))
                self.rpi_ip = rpi_ip
                print(f"✓ Connected to RPi for results at {rpi_ip}:{self.result_port}")
                
                # Start result sender thread
                result_thread = threading.Thread(target=self.send_results_thread)
                result_thread.daemon = True
                result_thread.start()
                
                return True
                
            except Exception as e:
                print(f"Failed to connect to RPi for results (attempt {attempt+1}): {e}")
                if self.result_socket:
                    self.result_socket.close()
                    self.result_socket = None
                time.sleep(2)
                
        return False
    
    def start(self):
        """Start the detection server"""
        print(f"Waiting for Raspberry Pi connection on port {self.video_port}")
        print("Color correction controls:")
        print("  Press '1' for AUTO mode")
        print("  Press '2' for BGR mode (RGB->BGR conversion)")
        print("  Press '3' for RGB mode (no conversion)")
        print("  Press '4' for NONE (no color correction)")
        print("  Press 's' to save screenshot")
        print("  Press 'q' to quit")
        
        try:
            conn, addr = self.video_socket.accept()
            rpi_ip = addr[0]
            print(f"Connected to Raspberry Pi at {rpi_ip}")
            
            # Start detection processing thread first
            detection_thread = threading.Thread(target=self.process_detection)
            detection_thread.daemon = True
            detection_thread.start()
            print("Detection processing thread started")
            
            # Start result connection in separate thread
            result_connect_thread = threading.Thread(target=self.connect_to_rpi_for_results, args=(rpi_ip,))
            result_connect_thread.daemon = True
            result_connect_thread.start()
            
            # Start video receiver (blocking)
            print("Starting video stream reception...")
            self.receive_video_stream(conn)
            
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detection server"""
        print("Stopping detection server...")
        self.running = False
        
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
                
        if self.result_socket:
            try:
                self.result_socket.close()
            except:
                pass
            
        cv2.destroyAllWindows()
        print("Detection server stopped")

if __name__ == "__main__":
    server = ObjectDetectionServer()
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping detection server...")
        server.stop()