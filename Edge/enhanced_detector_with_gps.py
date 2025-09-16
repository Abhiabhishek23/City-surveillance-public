import cv2
import threading
import queue
from geopy.geocoders import Nominatim
from edge_detector import EnhancedUrbanMonitor

class GeoAwareDetector(EnhancedUrbanMonitor):
    def __init__(self, camera_sources=None):
        super().__init__()
        self.geolocator = Nominatim(user_agent="city_surveillance")
        self.camera_locations = self.load_camera_locations()
        # camera_sources: camera_id -> source (0 for webcam, RTSP/HTTP URL)
        self.camera_sources = camera_sources or {"CAM_01": 0}
        self.frame_queues = {cam_id: queue.Queue(maxsize=1) for cam_id in self.camera_sources}

    def load_camera_locations(self):
        return {
            "CAM_01": {"coordinates": (23.1765, 75.7885), "location_ref": "Laptop Webcam"},
            "CAM_02": {"coordinates": (23.1780, 75.7900), "location_ref": "Mangalnath Ghat Area"},
            "CAM_03": {"coordinates": (23.1740, 75.7860), "location_ref": "Triveni Ghat Zone"}
        }

    def get_nearby_landmarks(self, coordinates):
        try:
            location = self.geolocator.reverse(f"{coordinates[0]}, {coordinates[1]}")
            if location and location.raw.get('address'):
                addr = location.raw['address']
                return {
                    "road": addr.get('road',''),
                    "suburb": addr.get('suburb',''),
                    "city": addr.get('city','Ujjain'),
                    "state": addr.get('state','MP')
                }
        except Exception as e:
            print(f"Geocoding error: {e}")
        return {"city":"Ujjain","state":"MP"}

    def send_alert_with_location(self, camera_id, event_type, frame, details=None):
        cam_data = self.camera_locations.get(camera_id, {})
        coordinates = cam_data.get("coordinates", (0,0))
        nearby_landmarks = self.get_nearby_landmarks(coordinates)
        enhanced_details = details or {}
        enhanced_details.update({
            "gps_coordinates": {"latitude": coordinates[0], "longitude": coordinates[1]},
            "location_reference": cam_data.get("location_ref","Unknown"),
            "nearby_landmarks": nearby_landmarks,
            "camera_id": camera_id
        })
        return super().send_alert(event_type, frame, enhanced_details)

    def process_frame(self, camera_id, frame):
        results = self.yolo_model(frame, classes=[0])
        people_boxes = results[0].boxes if results[0].boxes is not None else None
        people_count = len(people_boxes) if people_boxes else 0

        if people_count >= self.PERSON_LIMIT:
            self.send_alert_with_location(camera_id, "overcrowding", frame, {
                "count": people_count,
                "threshold": self.PERSON_LIMIT,
                "bounding_boxes": [box.xyxy[0].tolist() for box in people_boxes] if people_boxes else []
            })

        construction_detected, construction_masks, construction_boxes, construction_confidence = \
            self.detect_construction_mask_rcnn(frame)

        if not construction_detected:
            construction_detected, construction_contours = self.detect_construction_background_diff(frame)
        else:
            construction_contours = []

        if construction_detected:
            alert_details = {
                "method": "mask_rcnn" if construction_masks else "background_diff",
                "confidence": construction_confidence,
                "mask_count": len(construction_masks),
                "contour_count": len(construction_contours),
                "construction_areas": construction_boxes if construction_masks else []
            }
            self.send_alert_with_location(camera_id, "illegal_construction", frame, alert_details)

        visualized_frame = self.visualize_frame(
            frame, people_count, construction_detected,
            construction_masks, construction_contours
        )
        return visualized_frame

    def camera_thread(self, camera_id, source):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue  # keep trying
            # Keep only the latest frame in the queue
            if not self.frame_queues[camera_id].empty():
                try: self.frame_queues[camera_id].get_nowait()
                except: pass
            self.frame_queues[camera_id].put(frame)
        cap.release()

    def run(self):
        # Start threads for each camera (grab frames only)
        for cam_id, src in self.camera_sources.items():
            t = threading.Thread(target=self.camera_thread, args=(cam_id, src), daemon=True)
            t.start()

        # Main loop: process and display frames safely on main thread
        while True:
            for cam_id in self.camera_sources:
                if not self.frame_queues[cam_id].empty():
                    frame = self.frame_queues[cam_id].get()
                    vis_frame = self.process_frame(cam_id, frame)
                    cv2.imshow(f"{cam_id} Feed", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = {
        "CAM_01": 0,  # laptop webcam for testing
        # "CAM_02": "rtsp://192.168.1.100:554/stream1",
        # "CAM_03": "rtsp://192.168.1.101:554/stream1"
    }
    detector = GeoAwareDetector(camera_sources=cameras)
    detector.run()
