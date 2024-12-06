from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List
from PIL import Image
import cv2

from torch.utils.data.dataloader import DataLoader,Dataset
from facenet_pytorch.models.mtcnn import MTCNN

class VideoDataset(Dataset):
    '''
        Dataset class for fetching specific information about a video.
        The class requires a list of videos that it has to process.
    '''
    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        '''
            This function, picks a video and returns 4 values.
                str: Contains the name of the video
                list: List containing only the frame indices that are readable
                number: fps of the video
                list: a list containing all the frames as image arrays
        '''
        video = self.videos[index]
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(capture.get(5))
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
        return video, list(frames.keys()), fps, list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)
    
class VideoFaceDetector(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass

# Class implementing the MTCNN performing face detection
class FacenetDetector(VideoFaceDetector):

    def __init__(self, device="cuda:0") -> None:
        '''
            In this constructor, we define the MTCNN model. 
        '''
        super().__init__()

        '''
            detector will contain the MTCNN model. There are three parameters we pass to the detector.
            Threholds is the MTCNN face detection thresholds.
            Margin is the margin we add to the bounding box in terms of pixels in the final image. 
        '''
        self.detector =  MTCNN(
            device=device,
            thresholds=[0.85, 0.95, 0.95], 
            margin=0, 
        )

    def _detect_faces(self, frames) -> List:
        '''
            passes all the frames of a video and then returns the bboxes of each frame into batch_boxes. 
        '''
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        if batch_boxes is None:
            return []
        return [b.tolist() if b is not None else None for b in batch_boxes]

    @property
    def _batch_size(self):
        return 32

def detect_faces(video_path, device):

    '''
        video_path: str - Path to the video
        device: str - indicates whether to leverage CPU or GPU for processing
    '''

    '''
        We'll be using the facenet detector that is required to detect the faces present in each frame. This function is only responsible to return 
        a dictionary that contains the bounding boxes of each frame. 
        returns: 
            dict: dict template:
                {
                    frame_no: [[
                        [number, number, number, number],
                        [number, number, number, number],
                        ...
                        ...
                        [number, number, number, number]
                    ]]
                }
            int: fps of the video
    '''
    detector = FacenetDetector(device=device)

    
    # Read the video and its information
    dataset = VideoDataset([video_path])
    loader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1, collate_fn= lambda x : x)
    
    # Detect the faces
    for item in loader: 
        bboxes = {}
        video, indices, fps, frames = item[0]
        '''
            Update bboxes dict with the bounding boxes present in each frame with the frame number as the index and 
            a two dimensional list containing the bounding boxes as the value. 
        '''
        bboxes.update({i : b for i, b in zip(indices, detector._detect_faces(frames))})
        found_faces = False
        for key in list(bboxes.keys()):
            if isinstance(bboxes[key],list):
                found_faces = True
                break

        if not found_faces:
            return None, indices[-1]
    return bboxes,fps

def _get_frames(video_path):
    '''
        This function gets the video path, reads the video, stores the frames in a list and then returns
    '''
    
    # List to store the video frames
    frames = []
    
    # Read and store video Frames
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(5))

    for i in range(frames_num):
        capture.grab()
        success, frame = capture.retrieve()
        if not success:
            continue
        frames.append(frame)
    return frames

def _get_crop(frame, bbox, pad_constant: int | tuple):
    '''
    This function takes a frame and a bounding box and outputs the region of the image
    given by the bounding box with padding applied to all four sides.
    
    Args: 
        - frame (np.ndarray): The image frame containing the faces to be cropped.
        - bbox (list): The bounding box (xmin, ymin, xmax, ymax) for cropping.
        - pad_constant (int | tuple): The constant to control the padding.
          If an integer is provided, the same padding is applied in all directions.
          If a tuple (pad_w, pad_h) is provided, different padding is applied to width and height.

    Returns:
        - crop (np.ndarray): The cropped face region from the frame with padding.
    '''

    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
    w = xmax - xmin
    h = ymax - ymin

    # Handle padding logic
    if isinstance(pad_constant, int):
        p_w = p_h = pad_constant
    elif isinstance(pad_constant, tuple) and len(pad_constant) == 2:
        p_w, p_h = pad_constant
    else:
        raise ValueError("pad_constant should be either an int or a tuple of two values.")

    # Define padded crop area
    crop_ymin = max(ymin - p_h, 0)  # Ensure it doesn't go below 0
    crop_xmin = max(xmin - p_w, 0)  # Ensure it doesn't go left of 0
    crop_ymax = min(ymax + p_h, frame.shape[0])  # Ensure it doesn't go beyond image height
    crop_xmax = min(xmax + p_w, frame.shape[1])  # Ensure it doesn't go beyond image width

    # Extract the cropped region from the frame
    crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    return crop

# def _get_crop(frame, bbox, pad_constant : int | tuple):
#     '''
#         This function takes a frame and a bbox and then outputs the region of the image given by the bounding box
#         Args : 
#         - frame : np.ndarray -> image frame containing the faces to be cropped.
#         - bbox : list -> the bounding box associated with that frame.
#         - pad_constant : int -> The constant to control the padding. Default is None.
#         - use_pad_constant : bool -> If True, uses the pad_constant to control the padding. Default is False.

#         Returns :

#         - crop : np.ndarray -> the cropped output of the faces.
#     '''
#     xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
#     w = xmax - xmin
#     h = ymax - ymin

#     # Add some padding to catch background too
#     '''
#                           [[B,B,B,B,B,B],
#     [[F,F,F,F],            [B,F,F,F,F,B],
#      [F,F,F,F],    --->    [B,F,F,F,F,B],
#      [F,F,F,F]]            [B,F,F,F,F,B],
#                            [B,B,B,B,B,B]]

#             F -> Represents pixels with the Face.
#             B -> Represents the Background.
#             padding allows us to include some background around the face.
#             (padding constant 3 here causes some issue with some videos)
#     '''
#     p_w = 0
#     p_h = 0
#     if type(pad_constant) == int:
#         p_h = h // pad_constant
#         p_w = w // pad_constant
#     elif type(pad_constant) == float:
#         p_h = h // pad_constant[0]
#         p_w = w // pad_constant[1]

    
#     crop_h = (ymax + p_h) - max(ymin - p_h, 0)
#     crop_w = (xmax + p_w) - max(xmin - p_w, 0)

#     # Make the image square
#     '''
#     Makes the crop equal on all sides by adjusting the pad
#     '''
#     if crop_h > crop_w:
#         p_h -= int(((crop_h - crop_w)/2))
#     else:
#         p_w -= int(((crop_w - crop_h)/2))

#     # Extract the face from the frame
#     crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
    
#     # Check if out of bound and correct
#     h, w = crop.shape[:2]
#     if h > w:
#         diff = int((h - w)/2)
#         if diff > 0:         
#             crop = crop[diff:-diff,:]
#         else:
#             crop = crop[1:,:]
#     elif h < w:
#         diff = int((w - h)/2)
#         if diff > 0:
#             crop = crop[:,diff:-diff]
#         else:
#             crop = crop[:,:-1]

#     return crop

def extract_crops(video_path, bboxes_dict, pad_constant : int | tuple = 50):
    '''
    function that uses the above two function to extract faces and from individual frames

    Args:
     - video_path : str -> path to the video
     - bboxes_dict : dict -> dictionary containing the list of bounding boxes and their frame numbers as the key

        bboxes - > {
        frame_no(int) -> key :  list of bounding boxes(list(list(int)))
        }

        example -> {1 : [[45,689,5489,347],[474,543,434,454]],2 : [[435,435,222,321]]}

    - pad_constant : int | tuple-> controlling constant for padding.
    
    Returns:
     - crops : List[tuple] -> contains tuples with (frame_no(int), PIL Image of the cropped face, bbox(list(int)))
    
    '''
    frames = _get_frames(video_path)
    crops = []
    keys = [int(x) for x in list(bboxes_dict.keys())]
    for i in range(0, len(frames)):
        frame = frames[i]
        if i not in keys:
            continue
        bboxes = bboxes_dict[i]
        if not bboxes:
            continue
        for bbox in bboxes:
            crop = _get_crop(frame, bbox,pad_constant)
            fram = _get_crop(frame, bbox, 0)
            
            # Add the extracted face to the list
            crops.append((i, Image.fromarray(crop), bbox,Image.fromarray(fram)))

    return crops
