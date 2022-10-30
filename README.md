# DeepSort MaskRCNN - Track instance segmentation outputs

Recently I had a task which required tracking detections along with it's mask. I searched the internet for days to find anything that solves my problem but found no tutorial, articles, repository that solved my problem. <br/>

Then I found this repository: https://github.com/theAIGuysCode/yolov4-deepsort
Actually for object detecion there were many resources for tracking, so I chose the above one and decided to change it to tracking objects along with masks.

Tracking in DeepSort is 


## What exactly it will do?

This repository will guide you through tracking objects with masks produced by default MaskRCNN in Detectron2 framework.
You can train your custom MaskRCNN using Detectron2 easily following this tutorial:

Video 1: https://www.youtube.com/watch?v=ffTURA0JM1Q
Video2: https://www.youtube.com/watch?v=GoItxr16ae8&t=357s

Also here is another helpful repo: https://github.com/joheras/CLoDSA which will help with augmenting images with masks, in case you have limmited training data, thats what I used too.


## Usage

Open the Notebook given in this repo in Google Colab and run it following the instructions there, it's preety straightforward.
The file mask_tracker.py is the one that takes care of everything and the details of the codes are properly documented in the code, please follow it for clarification.


## Performance 

