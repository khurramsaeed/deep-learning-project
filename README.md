# Deep Learning Project (02456)
This project focus on the HELMET dataset and applying the YOLOv5 (You Only Look Once) model for realtime object detection.


The HELMET dataset contains 910 videoclips of motorcycle traffic, recorded at 12 observation sites in Myanmar in 2016. Each videoclip has a duration of 10 seconds, recorded with a framerate of 10fps and a resolution of 1920x1080. The dataset contains 10,006 individual motorcycles, surpassing the number of motorcycles available in existing datasets. Each motorcycle in the 91,000 annotated frames of the dataset is annotated with a bounding box, and rider number per motorcycle as well as position specific helmet use data is available. 

Can be accessed [here](https://osf.io/4pwj8/wiki/home/).

This repository holds the `notebook.ipynb` file which covers all of the data preperation steps in order to have dataset ready to be used with the YOLOv5 model.

### Steps to prepare the data
1. Download the data zip files from the link above.
2. Save them as siblings to `notebook.ipynb` file.
3. Run each code block `notebook.ipynb` (Note: One of the partx.zip has some meta data which corrupts the images while extracting)
4. If you have competed the above steps succesfully you will have a folder containing both the `images` and `labels` in the correct format.

### Run the model
```
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
%pip install -qr requirements.txt  # install

import torch
from yolov5 import utils

# Cd out
%cd ..

# Move the data fodler
!mv data yolov/data_yolo

# Move the traffic.yaml yolov5 folder
!mv traffic.yaml /yolov5

# Cd into YOLOv5 folder
!cd yolov5

# Run the training script
!python3 train.py --img 224 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 10 --epochs 10 --data traffic.yaml --weights yolov5s.pt  --name yolo_traffic
```

### Future work
1. Run the model with more epochs
2. Use the trained model with Nepal dataset
