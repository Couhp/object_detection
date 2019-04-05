# demo object detection by Imageai
from imageai.Detection import ObjectDetection
import os
from os import listdir
from os.path import join

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

def detect(detector, image_path, output_path):
    detector.detectObjectsFromImage(input_image=os.path.join(execution_path , image_path),
                                    output_image_path=os.path.join(execution_path , output_path))

img_list_name = listdir("image")
for img_name in img_list_name:
    image_path = join("image", img_name)
    output_path = join("result", img_name)
    detect(detector, image_path, output_path)
    


## FOR STORAGE
# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )