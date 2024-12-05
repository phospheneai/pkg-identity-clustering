
import os
from typing import List, Dict
import numpy as np
import json  
import cv2
from PIL import Image
from .base import _get_crop

def detect_probable_fakes(mask_frame, bbox, threshold = 0.50):
    mask = _get_crop(mask_frame,bbox, pad_constant=4)
    tot = mask.shape[0] * mask.shape[1]
    blob = np.sum(mask)
    if blob == 0.:
        return "Real"
    
    fake_prob = blob/tot
    if fake_prob >= threshold:
        return "Fake"
    else:
        return "Real"
    
def get_video_config(clustered_faces : Dict[int,list], identity : int, identity_dir : str | os.PathLike, mask_frames : List[np.ndarray] = None)->Dict[str,object]:
    config = {}
    config["ID"] = identity
    config["path"] = identity_dir
    config["class"] = "unassinged" 
    
    exact_crop_dir = os.path.join(identity_dir, "exact_crops")
    if not os.path.exists(exact_crop_dir):
        os.makedirs(exact_crop_dir)
    bboxes = []
    box = {}
    for frame, cropped_face, bbox, exact_crop in clustered_faces[identity]:
        if mask_frames and not config["class"]:
            config["class"] = detect_probable_fakes(mask_frames[frame],bbox) # type: ignore
        face_path = os.path.join(identity_dir,f"{frame}.jpg")
        image_cv = cv2.cvtColor(np.array(cropped_face), cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(image_cv)
        exact_crop_face_path = os.path.join(exact_crop_dir,f"{frame}.jpg")
        exact_crop = Image.fromarray(exact_crop)
        cropped_face.save(face_path)
        exact_crop.save(exact_crop_face_path)
        exact_crop.save()
        box[frame] = bbox
        with open(os.path.join(identity_dir,f"{frame}.json"),"w") as f:
            json.dump({frame : bbox},f)
    
    bboxes.append(box)
    config["bboxes"] = bboxes
    return config