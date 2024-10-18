from cluster import FaceCluster
from base import detect_faces, extract_crops
import torch
import os
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict



#(batc_size,frame)
class Inference():
    '''

    Class for Inferencing videos based on given model.

    '''
    def __init__(self, device : str, shape = (224,224)):
        '''
        Constructor for the inference class
        
        device (str) :->  the name of the device
        shape (Tuple(int)) :-> the fixed height and width of the clustered faces
        
        '''
        self.clusterer = FaceCluster()
        self.device = device
        self.shape = shape

    def _cvt_to_rgb(self, faces):
        '''
        function to convert BGR to RGB

        Args:

            faces Tuple(
                    int -(frame number),
                    PIL.Image.Image - (PIL Image),
                    List[float] - (Bbox)
                    ) :-> list of tuples containing frame number and images and bounding boxes
        
        Returns:
            List[Tensor(No.of.frames,height,width,channel)]
        
        '''
        out = []
        for i in faces:
            print(i[1])
            img = cv2.resize(np.array(i[1]),self.shape)
            out.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        out = torch.Tensor(out)
        return out

    def generate_video_data(self,video_path):
        '''
        function to generate list of identities for given videos

        Args: 
            video_path (str) -> full path to the video
        Returns :

            List[List[Tensor(No.of.frames,height,width,channel)]] -> list of identities
        '''

        if not os.path.exists(video_path):
            return []

        faces, fps = detect_faces(video_path, self.device)

        del fps

        crops = extract_crops(video_path,faces)
        clusters = self.clusterer.cluster_faces(crops)

        output = []
        for idx in list(clusters.keys()):
            
            output.append(self._cvt_to_rgb(clusters[idx]))

        return output


    def get_data(self, video_path):
        '''
        Returns a list of important video information

        Args:

        video_path (str) -> full path to the video

        Returns:
        Tuple(List[int], Dict[int,List[float]], List[PIL.Image.Image], int, Dict[int,List[Tuple(int,PIL.Image.Image,bbox)]]) --> list of frame numbers, faces, bounding boxes
        
        '''
        capture = cv2.VideoCapture(video_path)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = OrderedDict()
        for i in range(frames_num):
            capture.grab()
            success, frame = capture.retrieve()
            '''
                success will be false if the frame wasn't readable. We don't want 
                this in our ordered dict, so we'll check and move to next frame
                if it wasn't readable. 
            '''
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = frame.resize(size=[s // 2 for s in frame.size])
            frames[i] = frame

        faces , fps = detect_faces(video_path,self.device)
        crops = extract_crops(video_path,faces)
        clusters = self.clusterer.cluster_faces(crops)
        return list(frames.keys()), faces, list(frames.values()), fps, clusters


    def get_clusters(video_path : str | os.PathLike):
        pass
