import cv2
import numpy as np
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load custom YOLO model
model = YOLO(r"D:\Project_detection\Updated_code\validated_code\epoch20.pt")

# Initialize PaddleOCR for Tamil text detection
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# Set detection threshold
CONFIDENCE_THRESHOLD = 0.85  

def detect_edges_and_draw_contours(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Skipping empty or invalid frame")
            break  

        frame_count += 1
        results = model(frame)  # Run YOLO model inference

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf.item())  # Get confidence score
                print(f"confidenece score for the frame {frame_count} is {confidence}")
                if confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box

                    # Crop the detected bull region
                    bull_region = frame[y1:y2, x1:x2].copy()
                    
                    if bull_region.size > 0:
                        # Convert to grayscale and detect edges
                        gray_bull = cv2.cvtColor(bull_region, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray_bull, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        
                        # Find contours and create a mask
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        mask = np.zeros_like(gray_bull)
                        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

                        # Fill detected bull with red color
                        bull_colored = np.zeros_like(bull_region)
                        bull_colored[:, :] = (0, 0, 255)
                        bull_region[mask == 255] = bull_colored[mask == 255]
                        
                        # Replace the original region with processed region
                        frame[y1:y2, x1:x2] = bull_region
                        
                    print(f"Detected Bull in frame {frame_count} with {confidence*100:.2f}% confidence")

        # Define the bounding box coordinates (x, y, width, height)
        x, y, width, height = 1076, 727, 238, 56

       # Draw a white rectangle over the bounding box
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=-2)
        
        # Define text properties
        text = "Har Har Mahadev!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2

        # Put the text inside the masked rectangle
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                        
        out.write(frame)
        cv2.imshow("Bull Edge Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_video = r"D:\Project_detection\Updated_code\validated_code\indian-bull.mp4"
output_video = "output_bull_Final.avi"  
detect_edges_and_draw_contours(input_video, output_video)
