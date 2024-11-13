
import os
from typing import List, Dict
import numpy as np
import json  
import cv2
from PIL import Image
def get_video_config(clustered_faces : Dict[int,list], identity : int, identity_dir : str | os.PathLike, mask_frames : List[np.ndarray] = None)->Dict[str,object]:
    config = {}
    config["ID"] = identity
    config["path"] = identity_dir
    config["class"] = "" 
    
    
    bboxes = []
    box = {}
    for frame, cropped_face, bbox in clustered_faces[identity]:
        if not config["class"]:
            config["class"] = detect_probable_fakes(mask_frames[frame],bbox) # type: ignore
        face_path = os.path.join(identity_dir,f"{frame}.jpg")
        image_cv = cv2.cvtColor(np.array(cropped_face), cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(image_cv)
        cropped_face.save(face_path)
        box[frame] = bbox
        with open(os.path.join(identity_dir,f"{frame}.json"),"w") as f:
            json.dump({frame : bbox},f)
    
    bboxes.append(box)
    config["bboxes"] = bboxes
    return config