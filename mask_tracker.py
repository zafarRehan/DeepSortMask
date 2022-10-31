import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS




""" deep sort imports """
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from toolsTracker import generate_detections as gdet

""" image operations import """
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import matplotlib.pyplot as plt

""" import some common detectron2 utilities """
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
import matplotlib.colors as mplc


""" 
Initialize detectron2 maskRCNN model from detectron2 model zoo
if using other framework's model make sure to initialize it in same manner
along with it's respective imports

and dont remove 
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color

these 2 imports as they are used for drawing.
"""
cfg = get_cfg()
cfg.merge_from_file("/content/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80 
cfg.MODEL.WEIGHTS = os.path.join("detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the inference threshold for this model
predictor = DefaultPredictor(cfg)


""" command line flags definition """
flags.DEFINE_string('f', '', '')
flags.DEFINE_string('video', '', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '', 'path to output video')
flags.DEFINE_string('log_details', 'false', 'should log details to console or not. (true or false)')
flags.DEFINE_string('info_panel', 'true', 'should log details to information panel at top left corner or not. (true or false).\
                                 Slows down drawing process to some extent')

def main(_argv):

    """ initializing of tracking parameters """
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    """ initialize deep sort """
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    """ calculate cosine distance metric """
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    """ initialize tracker """
    tracker = Tracker(metric)
    
    video_path = FLAGS.video
    
    """ define a set of 25 colors for tracked objects on frame """
    colors = [mplc.to_rgb(random_color(rgb=True, maximum=1)) for c in range(25)]

   

    """ begin video capture """
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    """ position of top left corner dsplay block """
    x, y, w, h = 10, 10, int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)/4.5), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
    


    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"XVID")

    """ 
    Video output configuration:
    output path,
    codec, 
    fps of output video,
    width and height of output frames ( must match resultant frame )
    """
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    """
    list of all classes in the maskRCNN model in the same order as used for training the model.
    """
    class_names =  ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

    """
    list all classes that can be tracked by the tracker.
    If you want to skip certain classes from the model for tracking then
    exclude them from below list, else both class_names and allowed_classes
    will be same.
    """
    allowed_classes =  ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]

    frame_num = 0

    """ while video is running """
    while True:
        """
        Read frames from video
        """
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video Ended!!')
            break
        frame_num +=1
        print('frame_' + str(frame_num))
        start_time = time.time()

        """
        Perform prediction on frame and extract the following details:
            1. bounding boxes
            2. scores
            3. classes
            4. masks

        
        These are the infromation provided by all Instance Segmentation models.
        So, this code can be used to track on all Instance segmentation models.
        And, if an Instance Segmented video can be tracked then, Semantic Segmented
        videos can be tracked too with some tweaks in code.

        For maskRCNN models built with other frameworks can also be tracked with same code
        but for drawing on images detectron2 default functions are used so detectron2 will also
        be needed along with the other framework.
        """
        outputs = predictor(frame) # maskRCNN prediction in frame
        bboxes = outputs['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy() // 1
        scores = outputs['instances'].get_fields()['scores'].cpu().numpy()
        classes = outputs['instances'].get_fields()['pred_classes'].cpu().numpy()
        masks = outputs['instances'].get_fields()['pred_masks'].cpu().numpy()
        print('Detected: ', [class_names[clas] for clas in classes])
        num_objects = len(bboxes)


        """
        loop through objects and use class index to get class name, allow only classes in allowed_classes list
        """
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        
        """
        if there are predictions in the current frame from maskRCNN then change
        the bounding box format from (xmin, ymin, xmax, ymax) to (x, y, width, height)
        """
        temp_arr = []
        if(len(bboxes) > 0):
            for i in range(len(bboxes)):
                temp_arr.append([bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1]])
            bboxes = np.array(temp_arr)


        """ 
        frame and boxes aret sent to the model: /model_data/mars-small128.pb
        and it returns a set of features which will be used to determine
        tracked objects
        """
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, mask, feature) for bbox, score, class_name, mask, feature in zip(bboxes, scores, names, masks, features)]

        
        """
        Change all data to desired formats and then run Non Max Suppression to 
        filter out unwanted detections
        """
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        
        """ Call the tracker for tracked object predictions and updating new detections """
        tracker.predict()
        tracker.update(detections)

        white_rect = np.ones((h, w, 3), dtype=np.uint8) * 0
        
        if FLAGS.info_panel == 'true':
            """
            write text over the upper-left corner box which will later be merged to current frame after 
            drwaing the masks and boxes over it [number of objects is written once per frame]
            """
            cv2.putText(
                img = white_rect,
                text = "TRACKED OBJECTS: " + str(count),
                org = (20, 30),
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1.4,
                color = (255, 255, 255),
                thickness = 1
            )


        frame_masks = []
        frame_boxes = []
        frame_colors = []
        frame_labels = []
        # update tracks
        idx = 0
        """
        for each tracked object in current frame perform the following 
        operations.
        """
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
           
            bbox = track.to_tlbr()    # change bbox to (xmin, ymix, xmax, ymax, format)
            class_name = track.get_class()   # get class name from tracked object            
            mask = track.get_mask()   # get mask from the tracked object           
            score = ((track.get_score() * 10000) // 1) / 100      # get score and rescale it from 0 - 1 to 0 - 100         
            color = colors[int(track.track_id) % len(colors)]    # get color from colors list to maintain same 
                                                                 #color for same object accross frames
            
            """" append all values to lists to send for vizualization """
            frame_masks.append(mask)
            frame_boxes.append(bbox)
            frame_labels.append('id_' + str(track.track_id) + ' ' + class_name + ' ' + str(score))
            frame_colors.append(color)

            
            if FLAGS.info_panel == 'true':
                """
                write text over the upper-left corner box which will later be merged to current frame after 
                drwaing the masks and boxes over it [info is written n_objects per frame]
                """
                cv2.putText(
                    img = white_rect,
                    text = 'ID: ' + str(track.track_id) + '  ' + class_name + '  ' +  str([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]),
                    org = (20, 60 + (idx * 20)),
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1,
                    color = (color[2]*255, color[1]*255, color[0]*255),
                    thickness = 1
                )
                idx += 1


            """ if enabled log-details flag then print details about each track """
            if FLAGS.log_details == 'true':
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

           
        """
        If there are predictions on the current frame, then send it for 
        visualizations with the required data.
        """
        if(len(frame_colors) > 0):

            v = Visualizer(frame)
            v = v.overlay_instances (
              masks=np.array(frame_masks),
              boxes=frame_boxes,
              labels=frame_labels,
              keypoints=None,
              assigned_colors=frame_colors,
              alpha=0.3,
            )
            
            result = v.get_image()
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        else:
            result = np.asarray(frame)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


        """ calculate frames per second of running detections """
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        if FLAGS.info_panel == 'true':
            """
            merge the resultant frame with the top left corner info panel
            with some transparency 
            """
            sub_img = result[y:y+h, x:x+w]
            res = cv2.addWeighted(sub_img, 0.2, white_rect, 0.8, 1.0)
            result[y:y+h, x:x+w] = res
            
        out.write(result)         # save video file

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # app.run(main)
    try:
        app.run(main)
    except Exception as e:
        print("ERROR!, Something went wrong while trying to run the code.\nPlease refer to the following error msg" + repr(e))

