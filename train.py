from ultralytics import YOLO
import torch

def main():

    target_model=YOLO("yolov8m.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device)
    target_model.train(data="C:/dataset8/data.yaml",epochs=2000, imgsz=640, pretrained=True, optimizer="SGD", plots=True, cache='disk')


if __name__ == '__main__':
    main()
