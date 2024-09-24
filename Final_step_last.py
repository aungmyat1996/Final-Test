import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
from ultralytics import YOLO
import math
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("drone_control.log"),
        logging.StreamHandler()
    ]
)

class DroneController:
    def __init__(self, connection_string, model_path, takeoff_altitude=10, hover_timeout=5):
        self.connection_string = connection_string
        self.model_path = model_path
        self.takeoff_altitude = takeoff_altitude
        self.hover_timeout = hover_timeout
        self.vehicle = None
        self.model = None
        self.selected_target_bbox = None
        self.tracking_target = False
        self.search_running = False
        self.bounding_boxes = []
        self.target_type = 'person'
        self.last_target_time = time.time()
        self.lock = threading.Lock()
        self.search_thread = None
        self.setup()

    def setup(self):
        # Connect to the drone
        try:
            self.vehicle = connect(self.connection_string, wait_ready=True)
            logging.info("Connected to drone on %s", self.connection_string)
        except Exception as e:
            logging.error("Failed to connect to drone: %s", e)
            exit(1)
        
        # Initialize YOLO model with GPU support
        try:
            self.model = YOLO(self.model_path)
            if self.model.device.type != 'cpu':
                logging.info("YOLO model is using GPU: %s", self.model.device)
            else:
                logging.warning("YOLO model is using CPU. Consider using a GPU for better performance.")
            logging.info("YOLO model loaded successfully from %s", self.model_path)
        except Exception as e:
            logging.error("Failed to load YOLO model: %s", e)
            exit(1)

    def arm_and_takeoff(self):
        logging.info("Arming motors")
        while not self.vehicle.is_armable:
            logging.info("Waiting for vehicle to initialize...")
            time.sleep(1)
        
        logging.info("Taking off!")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True
        
        while not self.vehicle.armed:
            logging.info("Waiting for arming...")
            time.sleep(1)
        
        self.vehicle.simple_takeoff(self.takeoff_altitude)
        
        while True:
            current_altitude = self.vehicle.location.global_relative_frame.alt
            logging.info(f"Altitude: {current_altitude}")
            if current_altitude >= self.takeoff_altitude * 0.95:
                logging.info(f"Reached target altitude: {self.takeoff_altitude} meters")
                break
            time.sleep(1)

    def create_search_circle(self, center_location, radius, num_points=8):
        waypoints = []
        for i in range(num_points):
            angle = i * (360 / num_points)
            offset_x = radius * math.cos(math.radians(angle))
            offset_y = radius * math.sin(math.radians(angle))
            new_location = LocationGlobalRelative(
                center_location.lat + (offset_y / 111320),
                center_location.lon + (offset_x / (111320 * math.cos(math.radians(center_location.lat)))),
                center_location.alt
            )
            waypoints.append(new_location)
        return waypoints

    def is_same_target(self, bbox1, bbox2, threshold=50):
        x1, y1, x2, y2 = bbox1
        a1, b1, a2, b2 = bbox2
        center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
        center2 = ((a1 + a2) // 2, (b1 + b2) // 2)
        distance = math.hypot(center1[0] - center2[0], center1[1] - center2[1])
        return distance < threshold

    def start_search_pattern_async(self):
        with self.lock:
            if not self.search_running:
                self.search_thread = threading.Thread(target=self.start_search_pattern, daemon=True)
                self.search_thread.start()
                self.search_running = True
                logging.info("Started search pattern thread.")

    def start_search_pattern(self):
        try:
            center_location = self.vehicle.location.global_relative_frame
            search_waypoints = self.create_search_circle(center_location, radius=10)  # 10m radius

            for waypoint in search_waypoints:
                if not self.search_running:
                    break
                logging.info(f"Moving to waypoint: {waypoint}")
                self.vehicle.simple_goto(waypoint)
                time.sleep(5)  # Pause to allow the drone to reach each waypoint

        except Exception as e:
            logging.error(f"Error during search pattern: {e}")
        finally:
            with self.lock:
                self.search_running = False
                logging.info("Search pattern thread ended.")

    def process_yolo_detections(self, frame):
        results = self.model(frame)
        detections = results[0].boxes
        boxes = []

        for box in detections:
            cls = int(box.cls[0].item())
            cls_name = self.model.names[cls]
            if cls_name == self.target_type:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, cls_name))

        return boxes

    def move_drone_to_target(self, target_bbox):
        x1, y1, x2, y2 = target_bbox
        obj_x = (x1 + x2) // 2
        obj_y = (y1 + y2) // 2
        obj_width, obj_height = x2 - x1, y2 - y1
        frame_center_x, frame_center_y = 640 // 2, 480 // 2
        self.move_drone_based_on_position(obj_x, obj_y, obj_width, obj_height, frame_center_x, frame_center_y)

    def move_drone_based_on_position(self, obj_x, obj_y, obj_width, obj_height, frame_center_x, frame_center_y):
        x_offset = obj_x - frame_center_x
        y_offset = obj_y - frame_center_y

        x_proportion = abs(x_offset) / frame_center_x
        y_proportion = abs(y_offset) / frame_center_y

        # Yaw control (Channel 4)
        if abs(x_offset) > 20:
            if x_offset < 0:
                self.vehicle.channels.overrides['4'] = 1500 + int(300 * x_proportion)
            else:
                self.vehicle.channels.overrides['4'] = 1500 - int(300 * x_proportion)
        else:
            self.vehicle.channels.overrides['4'] = 1500

        # Pitch control (Channel 2)
        if abs(y_offset) > 20:
            if y_offset < 0:
                self.vehicle.channels.overrides['2'] = 1500 - int(300 * y_proportion)
            else:
                self.vehicle.channels.overrides['2'] = 1500 + int(300 * y_proportion)
        else:
            self.vehicle.channels.overrides['2'] = 1500

        # Altitude control (Channel 3)
        target_size = obj_width * obj_height
        altitude_adjustment = int((50000 - target_size) / 1000)
        if altitude_adjustment > 0:
            self.vehicle.channels.overrides['3'] = 1500 + min(altitude_adjustment, 200)
        else:
            self.vehicle.channels.overrides['3'] = 1500 - min(abs(altitude_adjustment), 200)

    def draw_bounding_boxes(self, frame, boxes, selected_bbox=None):
        for (x1, y1, x2, y2, cls_name) in boxes:
            if selected_bbox and self.is_same_target((x1, y1, x2, y2), selected_bbox):
                color = (0, 0, 255)  # Red for selected target
                label = "Following"
            else:
                color = (0, 255, 0)  # Green for unselected targets
                label = cls_name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display additional information
        battery = self.vehicle.battery
        if battery:
            battery_info = f"Battery: {battery.level}% | {battery.voltage}V"
        else:
            battery_info = "Battery: N/A"

        target_info = f"Target Type: {self.target_type.capitalize()}"
        tracking_info = "Tracking: ON" if self.tracking_target else "Tracking: OFF"
        search_info = "Search Running" if self.search_running else "Search: Idle"

        cv2.putText(frame, battery_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, target_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, tracking_info, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, search_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for (x1, y1, x2, y2, cls_name) in self.bounding_boxes:
                if x1 < x < x2 and y1 < y < y2 and cls_name == self.target_type:
                    self.selected_target_bbox = (x1, y1, x2, y2)
                    self.tracking_target = True
                    self.last_target_time = time.time()
                    logging.info(f"Selected target: {cls_name} at {self.selected_target_bbox}")
                    break

    def check_battery(self):
        if self.vehicle.battery is not None:
            voltage = self.vehicle.battery.voltage
            level = self.vehicle.battery.level  # Percentage
            logging.info(f"Battery Level: {level}% | Voltage: {voltage}V")
            if level is not None and level < 20:  # Threshold at 20%
                logging.warning("Battery low! Initiating emergency landing.")
                self.land()
        else:
            logging.warning("Battery information not available.")

    def land(self):
        logging.info("Landing the drone...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.mode.name != "LAND":
            logging.info("Waiting for LAND mode...")
            time.sleep(1)
        logging.info("Drone is landing.")
        # Wait until landed
        while self.vehicle.location.global_relative_frame.alt > 0:
            logging.info(f"Altitude during landing: {self.vehicle.location.global_relative_frame.alt}")
            time.sleep(1)
        logging.info("Drone has landed.")
        self.vehicle.close()
        logging.info("Drone connection closed.")
        exit(0)  # Exit the program

    def run(self):
        try:
            # Takeoff to the specified altitude
            self.arm_and_takeoff()

            cap = cv2.VideoCapture(0)  # Use camera feed or video file
            if not cap.isOpened():
                logging.error("Error: Could not open video source.")
                return

            cv2.namedWindow("Drone Target Tracking")
            cv2.setMouseCallback("Drone Target Tracking", self.mouse_callback)

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to grab frame.")
                    break

                # Resize frame for consistent processing (optional)
                frame = cv2.resize(frame, (640, 480))

                # Update bounding_boxes
                self.bounding_boxes = self.process_yolo_detections(frame)
                self.draw_bounding_boxes(frame, self.bounding_boxes, self.selected_target_bbox)

                if self.tracking_target and self.selected_target_bbox:
                    self.move_drone_to_target(self.selected_target_bbox)
                    self.last_target_time = time.time()  # Reset hover timer

                # If not tracking and hover timeout exceeded, start search
                if not self.tracking_target and (time.time() - self.last_target_time > self.hover_timeout):
                    self.start_search_pattern_async()

                # Check battery status periodically
                self.check_battery()

                cv2.imshow("Drone Target Tracking", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    self.target_type = 'person'
                    logging.info("Switched target type to: person")
                elif key == ord('c'):
                    self.target_type = 'car'
                    logging.info("Switched target type to: car")
                elif key == ord('e'):
                    logging.info("Emergency landing triggered by user.")
                    self.land()
                elif key == ord('h'):
                    logging.info("Hover command received. Hovering in place.")
                    self.vehicle.mode = VehicleMode("GUIDED")
                    self.vehicle.simple_goto(self.vehicle.location.global_relative_frame)
                elif key == ord('t'):
                    # Toggle target type between person and car
                    self.target_type = 'car' if self.target_type == 'person' else 'person'
                    logging.info(f"Toggled target type to: {self.target_type}")
                elif key == ord('s'):
                    if self.tracking_target:
                        self.tracking_target = False
                        logging.info("Tracking paused by user.")
                elif key == ord('r'):
                    if not self.tracking_target and self.selected_target_bbox:
                        self.tracking_target = True
                        self.last_target_time = time.time()
                        logging.info("Tracking resumed by user.")
                elif key == ord('q'):
                    logging.info("Quitting...")
                    break

                # Reset tracking if target is lost
                if self.tracking_target and not self.bounding_boxes:
                    self.tracking_target = False
                    self.selected_target_bbox = None
                    logging.info("Target lost. Hovering...")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            self.land()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.vehicle.mode.name != "LAND":
                self.land()
            logging.info("Drone disconnected and landed.")

if __name__ == "__main__":
    controller = DroneController(
        connection_string='udp:192.168.91.140:14550',
        model_path='yolov8n.pt',  # Using nano model for faster inference
        takeoff_altitude=10,  # meters
        hover_timeout=5  # seconds
    )
    controller.run()
