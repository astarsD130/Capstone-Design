import cv2
import socket
import pickle
import struct
import threading
import json
import time
import numpy as np
import serial
from picamera2 import Picamera2

class RPiCameraClient:
    def __init__(self, laptop_ip='192.168.43.168', video_port=8888, result_port=8889, 
                 serial_port='/dev/ttyUSB0', baudrate=115200):
        self.laptop_ip = laptop_ip
        self.video_port = video_port
        self.result_port = result_port
        
        # Serial communication with ESP32
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.esp_serial = None
        
        # Initialize camera - will be done in start() method
        self.picam2 = None
        
        # Frame dimensions for position calculation
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Position detection parameters
        self.position_threshold = 100  # pixels from center to trigger movement
        self.confidence_threshold = 0.5  # minimum confidence for detection
        
        # Socket for sending video
        self.video_socket = None
        
        # Socket for receiving detection results
        self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.result_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.result_socket.bind(('0.0.0.0', self.result_port))
        self.result_socket.listen(1)
        
        self.detection_results = []
        self.running = True
        self.last_command_time = 0
        self.command_cooldown = 0.5  # seconds between commands
        
    def initialize_serial(self):
        """Initialize serial communication with ESP32"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Initializing serial connection (attempt {attempt + 1}/{max_retries})...")
                
                # Try common serial ports
                serial_ports = [self.serial_port, '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
                
                for port in serial_ports:
                    try:
                        print(f"Trying port: {port}")
                        self.esp_serial = serial.Serial(
                            port=port,
                            baudrate=self.baudrate,
                            timeout=1,
                            write_timeout=1
                        )
                        
                        # Test communication
                        time.sleep(2)  # Wait for Arduino to reset
                        self.esp_serial.write(b"TEST\n")
                        self.esp_serial.flush()
                        
                        print(f"✓ Serial connection established on {port}")
                        return True
                        
                    except serial.SerialException as e:
                        print(f"Failed to connect to {port}: {e}")
                        if self.esp_serial:
                            self.esp_serial.close()
                            self.esp_serial = None
                        continue
                        
            except Exception as e:
                print(f"Serial initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        print("Failed to initialize serial connection")
        return False
    
    def send_command_to_esp(self, command):
        """Send movement command to ESP32"""
        if not self.esp_serial:
            return False
            
        # Implement command cooldown to prevent spam
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False
            
        try:
            command_str = f"{command}\n"
            self.esp_serial.write(command_str.encode())
            self.esp_serial.flush()
            self.last_command_time = current_time
            print(f"→ ESP32: {command}")
            return True
            
        except Exception as e:
            print(f"Error sending command to ESP32: {e}")
            return False
    
    def calculate_object_position(self, bbox):
        """Calculate object position relative to frame center"""
        x1, y1, x2, y2 = bbox
        
        # Calculate object center
        obj_center_x = (x1 + x2) // 2
        obj_center_y = (y1 + y2) // 2
        
        # Calculate distance from frame center
        dx = obj_center_x - self.frame_center_x
        dy = obj_center_y - self.frame_center_y
        
        # Determine position
        position = {
            'center_x': obj_center_x,
            'center_y': obj_center_y,
            'dx': dx,
            'dy': dy,
            'distance_from_center': np.sqrt(dx**2 + dy**2)
        }
        
        return position
    
    def determine_movement_command(self, detections):
        """Determine movement command based on object positions"""
        if not detections:
            return "STOP"
        
        # Filter detections by confidence
        valid_detections = [d for d in detections if d.get('confidence', 0) >= self.confidence_threshold]
        
        if not valid_detections:
            return "STOP"
        
        # Find the detection with highest confidence
        best_detection = max(valid_detections, key=lambda x: x.get('confidence', 0))
        
        # Calculate position
        bbox = best_detection['bbox']
        position = self.calculate_object_position(bbox)
        
        # Determine movement based on position
        dx = position['dx']
        dy = position['dy']
        
        # Check if object is close to center (no movement needed)
        if abs(dx) < self.position_threshold and abs(dy) < self.position_threshold:
            return "STOP"
        
        # Determine primary movement direction
        if abs(dx) > abs(dy):
            # Horizontal movement is more significant
            if dx > self.position_threshold:
                command = "RIGHT"  # Object is to the right, move right to follow
            elif dx < -self.position_threshold:
                command = "LEFT"   # Object is to the left, move left to follow
            else:
                command = "STOP"
        else:
            # Vertical movement is more significant
            if dy > self.position_threshold:
                command = "BACKWARD"  # Object is below center, move backward
            elif dy < -self.position_threshold:
                command = "FORWARD"   # Object is above center, move forward
            else:
                command = "STOP"
        
        # Log position info
        obj_class = best_detection.get('class', 'Unknown')
        confidence = best_detection.get('confidence', 0)
        print(f"Object: {obj_class} ({confidence:.2f}) at ({position['center_x']}, {position['center_y']}) -> {command}")
        
        return command
    
    def process_detections_for_movement(self, detections):
        """Process detection results and send movement commands"""
        try:
            # Determine movement command
            command = self.determine_movement_command(detections)
            
            # Send command to ESP32
            if self.esp_serial:
                self.send_command_to_esp(command)
            else:
                print(f"Serial not available. Would send: {command}")
                
        except Exception as e:
            print(f"Error processing detections for movement: {e}")

    def initialize_camera(self):
        """Initialize camera with proper error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Initializing camera (attempt {attempt + 1}/{max_retries})...")
                
                # Create new Picamera2 instance
                self.picam2 = Picamera2()
                
                # Configure with optimized settings
                video_config = self.picam2.create_video_configuration(
                    main={"size": (self.frame_width, self.frame_height), "format": "RGB888"},
                    lores={"size": (320, 240)}
                )
                self.picam2.configure(video_config)
                self.picam2.start()
                
                # Wait for camera to stabilize
                time.sleep(2)
                
                # Test capture
                test_frame = self.picam2.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    print(f"✓ Camera initialized successfully: {test_frame.shape}")
                    return True
                else:
                    raise Exception("Camera initialized but cannot capture frames")
                    
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                
                # Clean up on failure
                if self.picam2:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except:
                        pass
                    self.picam2 = None
                
                if attempt < max_retries - 1:
                    print("Waiting before retry...")
                    time.sleep(3)
                else:
                    print("Failed to initialize camera after all attempts")
                    return False
        
        return False
        
    def connect_to_laptop(self):
        """Connect to laptop for video streaming with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to laptop (attempt {attempt + 1}/{max_retries})...")
                self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.video_socket.settimeout(10.0)
                self.video_socket.connect((self.laptop_ip, self.video_port))
                print(f"✓ Connected to laptop at {self.laptop_ip}:{self.video_port}")
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if self.video_socket:
                    self.video_socket.close()
                    self.video_socket = None
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    print("Failed to connect after all attempts")
                    return False
    
    def send_video_stream(self):
        """Send video frames to laptop with proper synchronization"""
        if not self.picam2:
            print("Camera not initialized!")
            return
            
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        try:
            print("Starting video stream transmission...")
            
            while self.running:
                try:
                    # Capture frame from picamera2
                    frame = self.picam2.capture_array()
                    
                    if frame is None or frame.size == 0:
                        print(f"Frame {frame_count}: Empty frame captured")
                        continue
                    
                    # Convert from picamera2 format to BGR (OpenCV format)
                    if len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                        elif frame.shape[2] == 3:  # RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Validate frame
                    if frame is None or frame.size == 0:
                        print(f"Frame {frame_count}: Invalid frame after conversion")
                        continue
                    
                    # Encode frame as JPEG (direct binary data, not pickled)
                    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Slightly lower quality for stability
                    success, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if not success:
                        print(f"Frame {frame_count}: Failed to encode")
                        continue
                    
                    # Convert to bytes (this is what laptop expects)
                    frame_data = buffer.tobytes()
                    frame_size = len(frame_data)
                    
                    # Validate frame size (prevent corruption)
                    if frame_size > 1500000:  # 1.5MB limit
                        print(f"Frame {frame_count}: Size too large ({frame_size} bytes), skipping")
                        continue
                    
                    # Send frame size using network byte order (matches laptop expectation)
                    try:
                        size_bytes = struct.pack("!L", frame_size)  # Network byte order
                        self.video_socket.sendall(size_bytes)
                        
                        # Send frame data
                        self.video_socket.sendall(frame_data)
                        
                        frame_count += 1
                        fps_counter += 1
                        
                        # Print stats every 60 frames (reduce logging)
                        if frame_count % 60 == 0:
                            current_time = time.time()
                            elapsed = current_time - last_fps_time
                            if elapsed > 0:
                                fps = fps_counter / elapsed
                                print(f"Frame {frame_count}: Size={frame_size} bytes, FPS={fps:.1f}")
                                last_fps_time = current_time
                                fps_counter = 0
                        
                    except socket.error as e:
                        print(f"Socket error while sending frame {frame_count}: {e}")
                        break
                    except Exception as e:
                        print(f"Error sending frame {frame_count}: {e}")
                        break
                    
                    # Control frame rate (approximately 15 FPS for stability)
                    time.sleep(0.067)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    time.sleep(0.1)  # Brief pause on error
                    continue
                
        except KeyboardInterrupt:
            print("\nVideo streaming stopped by user")
        except Exception as e:
            print(f"Critical error in video streaming: {e}")
        finally:
            print(f"Video streaming ended. Total frames sent: {frame_count}")
    
    def receive_frame_data(self, conn, size):
        """Safely receive exact amount of data"""
        data = b""
        while len(data) < size:
            try:
                remaining = size - len(data)
                chunk_size = min(remaining, 8192)  # 8KB chunks
                packet = conn.recv(chunk_size)
                
                if not packet:
                    raise ConnectionError("Connection closed by laptop")
                    
                data += packet
            except socket.timeout:
                print("Timeout receiving results")
                raise
            except Exception as e:
                print(f"Error receiving result data: {e}")
                raise
                
        return data
    
    def receive_detection_results(self):
        """Receive detection results from laptop with better error handling"""
        try:
            print(f"Waiting for detection results connection on port {self.result_port}")
            conn, addr = self.result_socket.accept()
            conn.settimeout(30.0)  # Longer timeout
            print(f"✓ Detection result connection established from {addr}")
            
            result_count = 0
            
            while self.running:
                try:
                    # Receive data size (4 bytes) with network byte order
                    size_data = self.receive_frame_data(conn, 4)
                    size = struct.unpack("!L", size_data)[0]  # Network byte order
                    
                    if size > 1000000:  # 1MB limit for results
                        print(f"Result size too large: {size} bytes")
                        continue
                    
                    # Receive detection results
                    result_data = self.receive_frame_data(conn, size)
                    
                    # Deserialize results
                    results = pickle.loads(result_data)
                    self.detection_results = results
                    
                    result_count += 1
                    if result_count % 5 == 0:  # Reduce logging
                        print(f"Result {result_count}: Received {len(results)} detections ({size} bytes)")
                    
                    # Process detections for movement control
                    self.process_detections_for_movement(results)
                    
                    # Display results
                    self.display_results(results)
                    
                except ConnectionError:
                    print("Detection result connection lost")
                    break
                except socket.timeout:
                    print("No detection results received (timeout)")
                    continue
                except Exception as e:
                    print(f"Error receiving results: {e}")
                    time.sleep(1)
                    continue
                    
        except Exception as e:
            print(f"Error in result receiver: {e}")
        finally:
            print("Detection result receiver ended")
    
    def display_results(self, results):
        """Display detection results with position information"""
        if not results:
            return
            
        print(f"  Detection: {len(results)} objects found")
        for i, result in enumerate(results):
            class_name = result.get('class', 'Unknown')
            confidence = result.get('confidence', 0.0)
            bbox = result.get('bbox', [0, 0, 0, 0])
            
            # Calculate position
            position = self.calculate_object_position(bbox)
            pos_str = f"({position['center_x']}, {position['center_y']})"
            
            print(f"    {class_name}: {confidence:.2f} at {pos_str}")
    
    def test_connection(self):
        """Test connection to laptop before starting"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(3.0)
            test_socket.connect((self.laptop_ip, self.video_port))
            test_socket.close()
            print(f"✓ Connection test to {self.laptop_ip}:{self.video_port} successful")
            return True
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False
    
    def start(self):
        """Start the camera client with proper initialization"""
        print("=" * 50)
        print("Starting Raspberry Pi Camera Client with ESP32 Control")
        print("=" * 50)
        
        # Initialize serial communication first
        print("Initializing ESP32 serial communication...")
        if not self.initialize_serial():
            print("Warning: ESP32 serial connection failed. Commands will be logged only.")
        
        # Initialize camera
        if not self.initialize_camera():
            print("Camera initialization failed!")
            return
        
        # Test connection
        if not self.test_connection():
            print("Cannot reach laptop. Make sure laptop server is running first!")
            self.stop()
            return
        
        # Start result receiver thread FIRST
        print("Starting detection result receiver...")
        result_thread = threading.Thread(target=self.receive_detection_results)
        result_thread.daemon = True
        result_thread.start()
        
        # Wait for result thread to be ready
        time.sleep(1)
        
        # Connect to laptop for video
        if not self.connect_to_laptop():
            print("Failed to establish video connection")
            self.stop()
            return
        
        # Start video streaming (this is blocking)
        print("Starting video stream transmission...")
        print("Press Ctrl+C to stop")
        self.send_video_stream()
    
    def stop(self):
        """Stop the camera client safely"""
        print("\nStopping camera client...")
        self.running = False
        
        # Send stop command to ESP32
        if self.esp_serial:
            try:
                self.send_command_to_esp("STOP")
                time.sleep(0.1)
            except:
                pass
        
        try:
            if self.picam2:
                print("Stopping camera...")
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
                print("Camera stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")
        
        try:
            if self.esp_serial:
                self.esp_serial.close()
                print("ESP32 serial connection closed")
        except:
            pass
            
        try:
            if self.video_socket:
                self.video_socket.close()
                print("Video socket closed")
        except:
            pass
            
        try:
            if self.result_socket:
                self.result_socket.close()
                print("Result socket closed")
        except:
            pass
        
        print("Camera client stopped successfully")

if __name__ == "__main__":
    # Configuration
    LAPTOP_IP = "192.168.43.168"  # Update with your laptop's IP
    SERIAL_PORT = "/dev/ttyUSB0"   # Update with your ESP32 serial port
    BAUDRATE = 115200
    
    print("Raspberry Pi Object Detection Client with ESP32 Control")
    print("=" * 55)
    
    # Create client
    client = RPiCameraClient(
        laptop_ip=LAPTOP_IP,
        serial_port=SERIAL_PORT,
        baudrate=BAUDRATE
    )
    
    try:
        client.start()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        client.stop()
        print("Program ended")