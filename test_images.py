from ultralytics import YOLO
import os
def main():
   
    #%%
    model = YOLO(r"runs\detect\train40\weights\best.pt")
    
    
    test_path =r"C:\dataset8\test\images\\"
    test_images_dir = "train2/"
    test_results_dir = "results/"
    
    
    dimensions= (640,640)
    
    images = collect_images(test_images_dir)
    results= model.predict(images, conf= 0.505,imgsz=dimensions)
    
    i=0;
    
    for result in results:
        result.save(filename=test_results_dir+"result"+str(i)+".jpg")  
        i=i+1

 
def collect_images(datapath):
    images = []
    for filename in os.listdir(datapath):
        images.append(datapath + filename)
    return images

    #%%
if __name__ == '__main__':
    main()
