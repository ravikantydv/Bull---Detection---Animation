import cv2
import numpy as np
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
                print(f"Confidence score for frame {frame_count} is {confidence}")
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box
                    
                    # Zoom-out factor
                    zoom_out_factor = 1.1
                    
                    # Expand the bounding box
                    new_width = int((x2 - x1) * zoom_out_factor)
                    new_height = int((y2 - y1) * zoom_out_factor)
                    
                    x1 = max(0, x1 - (new_width - (x2 - x1)) // 2)
                    x2 = min(width, x1 + new_width)
                    y1 = max(0, y1 - (new_height - (y2 - y1)) // 2)
                    y2 = min(height, y1 + new_height)

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

                        # Apply red color to detected contours
                        bull_colored = np.zeros_like(bull_region)
                        bull_colored[:, :] = (0, 0, 255)
                        bull_region[mask == 255] = bull_colored[mask == 255]

                        # Resize to keep aspect ratio but not fill entire frame
                        aspect_ratio = bull_region.shape[1] / bull_region.shape[0]
                        new_width = int(width * 0.8)
                        new_height = int(new_width / aspect_ratio)
                        if new_height > height:
                            new_height = height
                            new_width = int(new_height * aspect_ratio)

                        zoomed_bull = cv2.resize(bull_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        
                        # Place the resized bull in the center of the frame
                        frame.fill(255)  # Make background white
                        x_offset = (width - new_width) // 2
                        y_offset = (height - new_height) // 2
                        frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = zoomed_bull

                    print(f"Detected Bull in frame {frame_count} with {confidence*100:.2f}% confidence")
        
        # Define the bounding box coordinates (x, y, width, height)
        x, y, width1, height1 = 1000, 750, 470, 100

        # Draw text overlay
        cv2.rectangle(frame, (50, height - 70), (350, height - 30), (0, 0, 0), thickness=-1)  # Black box

        cv2.rectangle(frame, (x, y), (x + width1, y + height1), (0, 0, 0), thickness=-2)

        # Define text properties
        text1 = "Har Har Mahadev!"
        font1 = cv2.FONT_HERSHEY_SIMPLEX
        font1_scale = 1
        font_thickness1 = 2
        text_size1 = cv2.getTextSize(text1, font1, font1_scale, font_thickness1)[0]
        text_x1 = x + (width1 - text_size1[0]) // 2
        text_y1 = y + (height1 + text_size1[1]) // 2


        # Put the text inside the masked rectangle
        cv2.putText(frame, text1, (text_x1, text_y1), font1, font1_scale, (255, 255, 255), font_thickness1)

        out.write(frame)
        cv2.imshow("Final Bull Zoomed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
        

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run the function
input_video = r"D:\Project_detection\Updated_code\validated_code\indian-bull.mp4"
output_video = "output_bull_zoomed_out.avi"  
detect_edges_and_draw_contours(input_video, output_video)
