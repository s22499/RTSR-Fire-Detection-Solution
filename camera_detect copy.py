from ultralytics import YOLO
import cv2
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fire dettection with Yolov8")
    parser.add_argument("--webcam-resolution", default=[1280,720], nargs=2, type=int)
    return parser.parse_args() 

def draw_bounding_boxes(frame, results, offset_x=0, offset_y=0):
    class_labels = {0: "fire", 1: "smoke"}  # Mapping of class ids to labels
    class_colors = {0: (0, 0, 255), 1: (255, 0, 0)}

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Get the bounding box coordinates and convert to numpy
                score = box.conf.item()
                class_id = box.cls.item()

                # Set the color for the bounding box
                color = class_colors[int(class_id)]

                # Draw the bounding box with specified color and thickness
                cv2.rectangle(frame, (int(x1) + offset_x, int(y1) + offset_y), (int(x2) + offset_x, int(y2) + offset_y), color, 3)

                # Add the label
                label = f"{class_labels[int(class_id)]}: {score:.2f}"
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                top_left = (int(x1) + offset_x, int(y1) - label_size[1] - base_line + offset_y)
                bottom_right = (int(x1) + label_size[0] + offset_x, int(y1) + offset_y)
                cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)

                # Draw the label text
                cv2.putText(frame, label, (int(x1) + offset_x, int(y1) - base_line + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, label, (int(x1) + offset_x, int(y1) - base_line + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return frame

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    print(frame_width, frame_height)
    model = YOLO(r"runs\detect\train43\weights\best.pt")

    cap = cv2.VideoCapture(1)
    actual_frame_width = cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    actual_frame_height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=0.403, stream=True, imgsz=1280)
        frame = draw_bounding_boxes(frame, results)
    
        cv2.imshow("frame", frame)
       
        if(cv2.waitKey(30)==27):
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()