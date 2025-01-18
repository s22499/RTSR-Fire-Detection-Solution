from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
def main():
    
    model = YOLO(r"runs\detect\train43\weights\best.pt")
    vidtest_dir = "vidtest/fire07.mp4"

    dimensions= (640,640)
    detections = []
    class_names = ["fire", "smoke"]
    frame_aggreration = 240

    for frame_idx, results in  enumerate(model.predict(source=vidtest_dir,stream=True, show=True, conf= 0.505,imgsz=dimensions)):
        
        smoke_count =0
        fire_count =0

        for box in results.boxes:
            class_id = int(box.cls[0])

            if class_names[class_id] == "fire":
                fire_count += 1
            elif class_names[class_id] == "smoke":
                smoke_count += 1
    
        frame_data = {
            'frame': frame_idx+1,
            'smoke': smoke_count,
            'fire': fire_count
        }
            
        detections.append(frame_data)
    
   
    draw_table(detections,frame_aggreration)
    
def draw_table(results, aggregation):
    
    aggregated_detections = []
    total_frames = len(results)

    for i in range(0, total_frames, aggregation):
        
        smoke_sum = sum([d['smoke'] for d in results[i:i+aggregation]])
        fire_sum = sum([d['fire'] for d in results[i:i+aggregation]])
        
        
        frame_label = results[min(i + aggregation - 1, total_frames - 1)]['frame']
        
        aggregated_detections.append({
            'frame': frame_label,
            'smoke': smoke_sum,
            'fire': fire_sum
        })


    frames = [d['frame'] for d in aggregated_detections]
    smoke_detections = [d['smoke'] for d in aggregated_detections]
    fire_detections = [d['fire'] for d in aggregated_detections]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    bar_width = 0.4
    index = np.arange(len(frames))
    
    
    smoke_bars = ax.bar(index, smoke_detections, bar_width, label='Smoke', color='blue')
    fire_bars = ax.bar(index + bar_width, fire_detections, bar_width, label='Fire', color='red')
    
    
    ax.set_title('Detections over Time/Frames')
    ax.set_xlabel('Frame/Time')
    ax.set_ylabel('Number of Detections')
    
    
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([str(f) for f in frames])
    
    
    ax.legend()
    
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
