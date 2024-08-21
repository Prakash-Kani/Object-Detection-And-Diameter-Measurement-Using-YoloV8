from ultralytics import YOLO
import cv2
import time


model = YOLO(r"D:\Prakash\Object_detections\model1\detect\train\weights\best.pt")  


def extract_predictions(results, confidence_threshold=0.5):
    predictions = []
    result = results[0].cpu()

    boxes = result.boxes.xywh.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    names = result.names

    for i in range(len(boxes)):
        if confidences[i] >= confidence_threshold:  # Apply the confidence threshold
            prediction = {
                'x': boxes[i][0],
                'y': boxes[i][1],
                'width': boxes[i][2],
                'height': boxes[i][3],
                'confidence': confidences[i],
                'class': names[int(class_ids[i])],
                'class_id': int(class_ids[i])
            }
            predictions.append(prediction)

    return {'predictions': predictions}


def process_frame(frame, predictions, scale_factor):
    # pred_diameter_mm_wi, pred_diameter_mm_wi, pred_diameter_mm_wi, pred_diameter_mm_wi = None, None, None, None
    for pred in predictions:
        x, y, width, height = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Calculate the coordinates for the bounding box
        start_point = (x - width // 2, y - height // 2)
        end_point = (x + width // 2, y + height // 2)
        
        # Draw the bounding box
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Create the label with class name and confidence
        label = f"{class_name}: {confidence:.2f}"
        
        # Put the label above the bounding box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y_min = max(start_point[1], label_size[1] + 10)
        cv2.rectangle(frame, (start_point[0], label_y_min - label_size[1] - 10), (start_point[0] + label_size[0], label_y_min + 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (start_point[0], label_y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        if class_name == 'Inner-Circle':
            diameter_px = width
            diameter_mm = diameter_px * scale_factor
            diameter_text = f"Diameter: {diameter_mm:.2f} mm"
            # Get text size for the diameter
            diameter_size, _ = cv2.getTextSize(diameter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            diameter_x = start_point[0]
            diameter_y = end_point[1] + 20
            
            # Draw rectangle for diameter text
            cv2.rectangle(frame, (diameter_x - 5, diameter_y - diameter_size[1] - 5), 
                    (diameter_x + diameter_size[0] + 5, diameter_y + 5), (176, 204, 227), cv2.FILLED)
        
            cv2.putText(frame, diameter_text, (start_point[0], end_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 36,18), 2)

            status = "Normal" if 65 <= diameter_mm <= 66.5 else "Defect"
            color = (0, 255, 0) if status == "Normal" else (18, 29, 255)
            
            # Get text size for status
            status_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            status_x = x - status_size[0] // 2
            status_y = y + status_size[1] // 2
            # Draw rectangle for status text
            cv2.rectangle(frame, (status_x - 5, status_y - status_size[1] - 10), 
                            (status_x + status_size[0] + 5, status_y + 5), (180, 233, 195), cv2.FILLED)
            cv2.putText(frame, status, (status_x , y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # Calculate the diameter in pixels
        
        # if class_name == 'Inner-Circle':
        #     diameter_px = width
        #     pred_diameter_mm_wi = diameter_px * scale_factor
        #     diameter_text = f"{pred_diameter_mm_wi:.2f} mm"
        #     # Get text size for the diameter
        #     diameter_size, _ = cv2.getTextSize(diameter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        #     diameter_x = start_point[0]
        #     diameter_y = end_point[1] + 20
            
        #     # Draw rectangle for diameter text
        #     cv2.rectangle(frame, (diameter_x - 5, diameter_y - diameter_size[1] - 5), 
        #             (diameter_x + diameter_size[0] + 5, diameter_y + 5), (176, 204, 227), cv2.FILLED)
        
        #     cv2.putText(frame, diameter_text, (start_point[0], end_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 36,18), 2)
        #     pred_diameter_mm_hi = height * scale_factor
        # if class_name == 'Outer-Circle':
        #     diameter_px = width
        #     pred_diameter_mm_wo = diameter_px * scale_factor
        #     diameter_text = f"{pred_diameter_mm_wo:.2f} mm"
        #     # Get text size for the diameter
        #     diameter_size, _ = cv2.getTextSize(diameter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        #     diameter_x = start_point[0]
        #     diameter_y = end_point[1] + 20
            
        #     # Draw rectangle for diameter text
        #     cv2.rectangle(frame, (diameter_x - 5, diameter_y - diameter_size[1] - 5), 
        #             (diameter_x + diameter_size[0] + 5, diameter_y + 5), (176, 204, 227), cv2.FILLED)
        
        #     cv2.putText(frame, diameter_text, (start_point[0], end_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 36,18), 2)
        #     pred_diameter_mm_ho = height * scale_factor
            

            
     
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame) 
        # image_()
        predictions = extract_predictions(results)
        frame_ = process_frame(frame, predictions['predictions'], scale_factor)


        cv2.imshow('Real-Time Object Detection', frame_)
        # time.sleep(10)
        print(f'Frame Count: {count}')
        
        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


known_diameter_mm = 48  # Known diameter of the reference object in millimeters
reference_diameter_px = 277  # Diameter of the reference object in pixels (measure manually if needed)
scale_factor = known_diameter_mm / reference_diameter_px


main()
