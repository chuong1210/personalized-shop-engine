import cv2
import numpy as np

class ROISelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.roi_points = []
        self.window_name = "ROI Selection - Press R to start, Enter to finish, ESC to cancel"
        self.selecting = False
        self.instruction_text = "Press 'R' to start selecting ROI"
        
    def mouse_callback(self, event, x, y, flags, param):
        if not self.selecting:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))
            self.display_frame = self.frame.copy()
            
            # Draw all points
            for i, pt in enumerate(self.roi_points):
                cv2.circle(self.display_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(
                    self.display_frame, f"P{i+1}", 
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
            
            # Draw lines
            if len(self.roi_points) > 1:
                for i in range(len(self.roi_points) - 1):
                    cv2.line(
                        self.display_frame,
                        self.roi_points[i],
                        self.roi_points[i + 1],
                        (0, 255, 0), 2
                    )
            
            self.instruction_text = f"{len(self.roi_points)} points selected. Left click to add more, Enter to finish"
    
    def draw_instructions(self):
        # Background for text
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (self.display_frame.shape[1] - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)
        
        # Instructions
        cv2.putText(
            self.display_frame, self.instruction_text,
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            self.display_frame, "Controls: R=Start | Enter=Finish | ESC=Cancel | Left Click=Add Point",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
    
    def select_roi(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            self.draw_instructions()
            cv2.imshow(self.window_name, self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r') or key == ord('R'):
                if not self.selecting:
                    self.selecting = True
                    self.roi_points = []
                    self.instruction_text = "Left click to add ROI points. Press Enter when done."
            
            elif key == 13:  # Enter
                if len(self.roi_points) >= 3:
                    # Close polygon
                    self.display_frame = self.frame.copy()
                    roi_array = np.array(self.roi_points, np.int32)
                    cv2.polylines(self.display_frame, [roi_array], True, (0, 255, 0), 3)
                    for pt in self.roi_points:
                        cv2.circle(self.display_frame, pt, 5, (0, 0, 255), -1)
                    
                    self.instruction_text = f"ROI with {len(self.roi_points)} points selected! Closing..."
                    self.draw_instructions()
                    cv2.imshow(self.window_name, self.display_frame)
                    cv2.waitKey(1000)
                    break
                else:
                    self.instruction_text = "Need at least 3 points! Continue clicking."
            
            elif key == 27:  # ESC
                self.roi_points = []
                self.instruction_text = "ROI selection cancelled"
                self.draw_instructions()
                cv2.imshow(self.window_name, self.display_frame)
                cv2.waitKey(1000)
                break
        
        cv2.destroyWindow(self.window_name)
        return np.array(self.roi_points, np.int32) if len(self.roi_points) >= 3 else None


class ScaleCalibrator:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []
        self.window_name = "Scale Calibration - Press C to start, Select 2 points"
        self.selecting = False
        self.instruction_text = "Press 'C' to start calibration"
        
    def mouse_callback(self, event, x, y, flags, param):
        if not self.selecting or len(self.points) >= 2:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.display_frame = self.frame.copy()
            
            # Draw points
            for i, pt in enumerate(self.points):
                cv2.circle(self.display_frame, pt, 7, (0, 0, 255), -1)
                cv2.putText(
                    self.display_frame, f"P{i+1}",
                    (pt[0] + 15, pt[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
            
            # Draw line if 2 points
            if len(self.points) == 2:
                cv2.line(self.display_frame, self.points[0], self.points[1], (0, 255, 0), 3)
                pixel_dist = np.linalg.norm(
                    np.array(self.points[1]) - np.array(self.points[0])
                )
                self.instruction_text = f"Distance: {pixel_dist:.1f} pixels. Press Enter to input real distance"
    
    def draw_instructions(self):
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (self.display_frame.shape[1] - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)
        
        cv2.putText(
            self.display_frame, self.instruction_text,
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            self.display_frame, "Controls: C=Start | Left Click=Add Point | Enter=Finish | ESC=Cancel",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
    
    def calibrate(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            self.draw_instructions()
            cv2.imshow(self.window_name, self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                if not self.selecting:
                    self.selecting = True
                    self.points = []
                    self.instruction_text = "Click 2 points with known real-world distance"
            
            elif key == 13 and len(self.points) == 2:  # Enter
                break
            
            elif key == 27:  # ESC
                self.points = []
                break
        
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) == 2:
            pixel_dist = np.linalg.norm(
                np.array(self.points[1]) - np.array(self.points[0])
            )
            return pixel_dist
        return None