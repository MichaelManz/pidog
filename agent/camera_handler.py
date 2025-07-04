import cv2
import time
import base64
from vilib import Vilib


class CameraHandler:
    """Handles camera initialization, image capture, and cleanup operations."""
    
    def __init__(self, vflip=False, hflip=False, local_display=False, web_display=False):
        """
        Initialize camera handler with specified settings.
        
        Args:
            vflip: Vertical flip the camera image
            hflip: Horizontal flip the camera image
            local_display: Enable local display
            web_display: Enable web display
        """
        self.vflip = vflip
        self.hflip = hflip
        self.local_display = local_display
        self.web_display = web_display
        self.is_started = False
        
    def start(self):
        """Start the camera and display services."""
        if self.is_started:
            return
            
        Vilib.camera_start(vflip=self.vflip, hflip=self.hflip)
        Vilib.display(local=self.local_display, web=self.web_display)
        
        # Wait for flask to start
        while True and self.web_display:
            if Vilib.flask_start:
                break
            time.sleep(0.01)
        
        time.sleep(0.5)
        self.is_started = True
        print('\nCamera started successfully')
        
    def capture_image(self, save_path='./img_input.jpg'):
        """
        Capture current image from camera.
        
        Args:
            save_path: Path to save the captured image
            
        Returns:
            str: Path to the saved image file
        """
        if not self.is_started:
            raise RuntimeError("Camera not started. Call start() first.")
            
        cv2.imwrite(save_path, Vilib.img)
        return save_path
        
    def get_image_base64(self, save_path='./img_input.jpg'):
        """
        Capture image and return as base64 encoded string.
        
        Args:
            save_path: Temporary path to save image before encoding
            
        Returns:
            str: Base64 encoded image data
        """
        image_path = self.capture_image(save_path)
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        return image_data
        
    def get_current_image(self):
        """
        Get the current image from camera without saving to file.
        
        Returns:
            numpy.ndarray: Current camera image
        """
        if not self.is_started:
            raise RuntimeError("Camera not started. Call start() first.")
            
        return Vilib.img
        
    def close(self):
        """Close the camera and cleanup resources."""
        if self.is_started:
            Vilib.camera_close()
            self.is_started = False
            print("Camera closed") 