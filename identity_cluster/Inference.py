from cluster import FaceCluster
from base import detect_faces, extract_crops
import torch
import os
import cv2
import numpy as np
from PIL import Image
from typing import List
from collections import OrderedDict
from models.models_list import ModelList
import time
import matplotlib.pyplot as plt


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
        self.classes = ["Real","Fake"]
        

        #The below cluster variable has to be verified.
        self._clusters = None
    def __cvt_to_rgb(self, faces):
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
            img = cv2.resize(np.array(i[1]),self.shape)
            out.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        out = torch.Tensor(out)
        return out
    def __plot_images_grid(self,tensor, images_per_row=4):
        """
        Plots a grid of images from a 4D tensor.
        
        Args:
            tensor (torch.Tensor): A 4D tensor of shape (N, H, W, C), (N, C, H, W), or similar.
            images_per_row (int): Number of images per row in the grid.
        """
        # Check tensor dimensions and permute if necessary
        if tensor.dim() == 4:
            if tensor.size(1) in [1, 3] and tensor.size(2) > 1 and tensor.size(3) > 1:  # (N, C, H, W)
                images_np = tensor.permute(0, 2, 3, 1).numpy()  # Convert to (N, H, W, C)
            elif tensor.size(3) in [1, 3] and tensor.size(1) > 1 and tensor.size(2) > 1:  # (N, H, W, C)
                images_np = tensor.numpy()  # Already in (N, H, W, C)
            else:
                raise ValueError("Tensor dimensions do not match expected shapes for images.")
        else:
            raise ValueError("The input tensor must be 4D with shape (N, C, H, W) or (N, H, W, C).")

        # Normalize images if necessary (assuming images are in the range [0, 1] or [0, 255])
        if images_np.max() > 1.0:  # If max value is greater than 1, assume range is [0, 255]
            images_np = images_np / 255.0

        # Handle grayscale images by converting them to RGB
        if images_np.shape[-1] == 1:
            images_np = np.repeat(images_np, 3, axis=-1)  # Convert single channel to 3 channels

        num_images = images_np.shape[0]
        num_rows = (num_images + images_per_row - 1) // images_per_row  # Calculate number of rows needed

        # Create a figure and axes
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 2, num_rows * 2))
        
        # Flatten the axes array for easy indexing
        axes = axes.flatten()

        # Plot each image
        for i in range(num_images):
            ax = axes[i]
            ax.imshow(images_np[i])
            ax.axis('off')  # Hide axis

        # Hide any unused subplots
        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    def __print_result(self, result : dict, image_data : List[torch.Tensor]):

        for i in result:
            curr = result[i]["class"]
            print(f"Identity : {curr}")
            self.__plot_images_grid(image_data[i][:5])

    def __create_sequence_dict(self,identity_data):
        sequence_dict = {}

        for identity_id, identity_info in identity_data.items():
            for data in identity_info['data']:
                frame_idx = data[0]
                face_bboxes = data[-1]
                # frame_idx, pil,face_bboxes 
                if frame_idx not in sequence_dict:
                    sequence_dict[frame_idx] = []
                
                # for bbox in face_bboxes:
                entry = {
                    "data": face_bboxes,
                    "class": identity_info["class"],
                    "confidence":identity_info["confidence"]
                }
                sequence_dict[frame_idx].append(entry)

        return sequence_dict
    def __draw_bounding_boxes(self,video_path : str, sequence_dict, result_video_path : str | os.PathLike) -> str:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(5))
        output = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc("X", "V", "I", "D"), fps, (width, height))

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if there are any bounding boxes for the current frame index
            if frame_index in sequence_dict:
                for item in sequence_dict[frame_index]:
                    bbox = item['data']
                    class_label = item['class']
                    color = (0,255,0) if class_label=='real'   else (0,0,255) 
                    confidence = round(item["confidence"]*100,2)
                        
                
                    xmin, ymin, xmax, ymax = map(int, [b*2 for b in bbox])

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    cv2.putText(frame, f"{class_label} {confidence}%"  , (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Write the processed frame to the output video
            output.write(frame)
            frame_index += 1

        output.release()
        cap.release()
        print("Video processing complete. Output saved to:", result_video_path)

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
            
            output.append(self.__cvt_to_rgb(clusters[idx]))
        self._clusters = clusters

        return output, len(clusters.keys())


    def get_data(self, video_path):
        '''
        Returns a list of important video information

        Args:

        video_path (str) -> full path to the video

        Returns:
        Tuple(List[int], Dict[int,List[float]], List[PIL.Image.Image], int, Dict[int,List[Tuple(int,PIL.Image.Image,bbox)]]) --> 
        List[int] -> list frame numbers
        Dict[int,List[List[float]]] -> dict of bounding boxes for each identity in each frame
        List[PIL.Image.Image] -> list of images in order of frame number and identity present in each frame in the dictionary object returned as par of the return statement which is mentioned above
        int -> fps of the video
        Dict[int,List[Tuple(int,PIL.Image.Image,bbox)]] -> dict of clustered identities.
        
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

    def get_predictions(self, model,images, device='cuda', print_time = True):
        '''
        The function that takes in a model object, a tensor of all identities in the 
        '''
        start_time = time.time()
        images = images.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device)
        images=images/255.
        print(images.shape)
        with torch.no_grad():
            logits = model(images)
            predicted_labels = torch.argmax(logits, dim=1)
            
            results = {
                'logits': logits,
                'predicted_labels': predicted_labels.item()
            }
        end_time = time.time()
        if print_time == True:
            print(f"Execution time: {end_time - start_time} seconds")
        
        return results
    def inference_models(self, video_path : str | os.PathLike | List[str], model_name : str, model_weights_path : str | os.PathLike, save_result_vid : bool = False, save_path : str | None = None):
        '''
        Function for inferencing models from deepcheck model zoo for a list of videos

        Args:

            video_path (str | os.PathLike | List[str]) -> full path to the video or list of videos
            model_name (str) -> name of the deepcheck model
            model_weights_path (str | os.PathLike) -> full path to the model weights
        
        Returns:

        List[Dict[str, Any]] -> list of dictionaries containing prediction results for each video
        '''

        if not ModelList.get_model(model_name,model_weights_path, self.device):
            return "There is no such model"
        
        model = ModelList.get_model(model_name, model_weights_path, self.device)
        model.eval()

        inp = None
        identities = None
        if isinstance(video_path, str) or isinstance(video_path,os.PathLike):
            inp, identities = self.generate_video_data(video_path)
        if inp and identities:
            res = {}
            for i in range(identities):
                preds = self.get_predictions(model,inp[i])
                res[i] = {}
                res[i]["class"] = self.classes[preds["predicted_labels"]]
                res[i]["confidence"] = torch.max(preds["logits"]).cpu().item()
                res[i]["data"] = self._clusters[i]
                print(self._clusters)
            self.__print_result(res,inp)
            if save_result_vid:
                seq_dict = self.__create_sequence_dict(res)
                self.__draw_bounding_boxes(video_path,seq_dict,save_path)
            return res
