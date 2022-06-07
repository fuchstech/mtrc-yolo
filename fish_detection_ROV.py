#!/usr/bin/env python
"""
BlueRov video capture class
"""

import cv2
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
        latest_frame (np.ndarray): Latest retrieved video frame
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self.latest_frame = self._new_frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps_structure = sample.get_caps().get_structure(0)
        array = np.ndarray(
            (
                caps_structure.get_value('height'),
                caps_structure.get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            np.ndarray: latest retrieved image frame
        """
        if self.frame_available:
            self.latest_frame = self._new_frame
            # reset to indicate latest frame has been 'consumed'
            self._new_frame = None
        return self.latest_frame

    def frame_available(self):
        """Check if a new frame is available

        Returns:
            bool: true if a new frame is available
        """
        return self._new_frame is not None

    def run(self):
        """ Get frame to update _new_frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        self._new_frame = self.gst_to_opencv(sample)

        return Gst.FlowReturn.OK



# Create the video object
# Add port= if is necessary to use a different one
video = Video()

print('Initialising stream...')
waited = 0
while not video.frame_available():
    waited += 1
    print('\r  Frame not available (x{})'.format(waited), end='')
    cv2.waitKey(30)
print('\nSuccess!\nStarting streaming - press "q" to quit.')

while True:
    frame = video.frame()
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)

    
    labels = ["Balik"]
#%%
    
    colors = ["0,0,255", "0,255,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))
    
    model = cv2.dnn.readNetFromDarknet("/home/fuchs/Desktop/Darknet/darknet/fish_detection/yolov3-obj.cfg", "/home/fuchs/Desktop/Darknet/darknet/yolov3-obj_30000.weights?dl=0")
    
    layers = model.getLayerNames()
    
    output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)

    
    ids_list = []
    boxes_list = []
    confidences_list = []




    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.4:
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4]*np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x-(box_width/2))
                start_y = int(box_center_y-(box_height/2))
                
                ##NON MAXIMUM SUPPRESSION OPer   ation 2 start
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                 
                ###END SUPPRESSION  OPeration 2 end
                
##NON MAXIMUM SUPPRESSION  OPeration 3 start
            
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]
        
        start_x = box[0]
        start_x = box[1]
        box_width = box[2]
        box_height = box[3]
        
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
        
    
     ###END SUPPRESSION  OPeration 3 end
                
                
                
        end_x = start_x + box_width
        end_y = start_y + box_height
        
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
        
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(frame, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    cv2.namedWindow("Detection Window")
    cv2.resizeWindow('image', 1200, 900) 
    cv2.imshow("Detection Window", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()

            
