import cv2

class USBCamera:
    def __init__(self, device_index=0):
        self.device_index=device_index
        self.cap=cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera(index={self.device_index}) could not opened")
        
    def get_picture(self):
        ret, frame=self.cap.read()

        if not ret:
            raise RuntimeError("Frame could not read")
        
        return frame
    
    def release(self):
        self.cap.release()