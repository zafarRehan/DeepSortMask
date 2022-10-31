# DeepSort MaskRCNN - Track instance segmentation outputs

Recently I had a task which required tracking detections along with it's mask. I searched the internet for days to find anything that solves my problem but found no tutorial, articles, repository that solved my problem. <br/>

Then I found this repository: https://github.com/theAIGuysCode/yolov4-deepsort
Actually for object detecion there were many resources for tracking, so I chose the above one and decided to change it to tracking objects along with masks.

Tracking in DeepSort is 


## What exactly it will do?

This repository will guide you through tracking objects with masks produced by default MaskRCNN in Detectron2 framework.
You can train your custom MaskRCNN using Detectron2 easily following this tutorial:

Video 1: https://www.youtube.com/watch?v=ffTURA0JM1Q <br/>
Video2: https://www.youtube.com/watch?v=GoItxr16ae8&t=357s

Also here is another helpful repo: https://github.com/joheras/CLoDSA which will help with augmenting images with masks, in case you have limited training data, thats what I used too.


## Usage

Open the <a href="maskRCNN_tracking.ipynb">Notebook</a> given in this repo in Google Colab and run it following the instructions there, it's preety straightforward.
The file <a href="mask_tracker.py">mask_tracker.py</a> is the one that takes care of everything and the details of the codes are properly documented in the code, please follow it for clarification.


## Performance 

### Input
Input videos are there in <a href="/data/video/">/data/video/</a>


### Output
<a href="https://drive.google.com/file/d/1Gf8NUKqZJ2PN4hhAEPdQ-oQ2PPK8lN0v/view?usp=share_link"><img src="/cars.png" width=800></a>
<br/><br/><br/>
<a href="https://drive.google.com/file/d/1fP_wjmwxBWg1R6Ij7QHbaL4aHFvn83es/view?usp=share_link"><img src="/public.png" width=800></a>



## Conclusion

This repo guides you with tracking masked objects using Detectron2 MaskRCNN in detail. This same method can be used to track other model's Object Detections or Segmentations in different architectures too, some tweaks would be needed, try to figure it out and if some help is needed can raise an Issue or connect with me on linkedIn https://www.linkedin.com/in/rehan-zafar-48797b193/

If this repository helped you in anyway please give it a star.

Thanks,
REHAN ZAFAR
