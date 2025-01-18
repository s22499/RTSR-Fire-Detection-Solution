from ultralytics import YOLO
import torch

def main():
    dimensions =(640,640)
    model = YOLO(r"runs\detect\train40\weights\best.pt")
    model.val(data="C:/dataset8/data.yaml",imgsz=dimensions, conf=0.505)

if __name__ == '__main__':
    main()
