import networkx as nx
import numpy as np

import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision.transforms import Resize
from typing import List, Dict
from .base import extract_crops

class FaceCluster:

    def __init__(self, crops = None, similarity_threshold : int = 0.85,
                 device : str = "cpu"
                 ):
        self.similarity_threshold = similarity_threshold

        self.device = torch.device(device)

        self.crops = crops

    def _set_crops(self,crops):
        '''
        A setter function to set the threshold attribute
        Args :
         - crops : List[tuple] -> contains tuples with (frame_no(int), PIL Image of the cropped face, bbox(list(int)))

        Returns :
            None
        '''
        self.crops = crops

    def _set_threshold(self,threshold):
        '''
        A setter function to set the threshold attribute
        Args :
         - threshold : float -> threshold value.

        Returns :
            None
        '''
        self.similarity_threshold = threshold
    
    def _preprocess_images(self, img, shape=[128, 128]):
        '''
        function to resize the image

        Args : 
        - img : np.ndarray -> img to be resized
        - shape : list -> the shape of the output image

        Returns :
         - img : np.ndarray -> resized image
        '''
        img = Resize(shape)(img)
        return img

    def _generate_connected_components(self, similarities):
        '''
        Helper function for the clustering function, takes in dot product similarities and clusters them based on a predefined threshold,
        the threshold can be set by the user when intializing the class or can be set while calling the `cluster_faces` function
        
        Args :
         - similarities : np.ndarray -> similarity matrix (attention without scaling)

        Returns :
         - components : list -> list of clustered faces
        '''
        graph = nx.Graph()
        for i in range(len(similarities)):
            for j in range(len(similarities)):
                # take every face dot product value with every other value except itself and compare with threshold, when 
                #if the value is greater than the threshold thhen add an edge between them to signify that they are the same face.
                if i != j and similarities[i, j] > self.similarity_threshold:
                    graph.add_edge(i, j)
        #get all the clustered components and add them to the resultant component list
        components_list = []
        for component in nx.connected_components(graph):
            components_list.append(list(component))
        #for memory optimization clear the graph that was created
        graph.clear()
        graph = None

        return components_list
    
    def cluster_faces(self,crops, threshold = None):
        '''
        function that clusters the faces using the dot product similarity metric.

        Args:
         - crops : List[tuple] -> contains tuples with (frame_no(int), PIL Image of the cropped face, bbox(list(int)))
         - threshold : Optional[float] -> set to change the threshold attribute.
        
        Returns :
         - clustered_faces : Dict[int,list] -> returns a dictionary containing the identity as keys and all the faces associated with a single
                                                identity as a list.
        
        '''
        if threshold:
            self._set_threshold(threshold)
        
        if crops:
            self._set_crops(crops)

    # Convert crops to PIL images
        crops_images = [row[1] for row in self.crops]
        
        # Extract the embeddings
        embeddings_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        faces = [self._preprocess_images(face) for face in crops_images]
        faces = np.stack([np.uint8(face) for face in faces])
        faces = torch.as_tensor(faces)
        faces = faces.permute(0, 3, 1, 2).float()
        faces = fixed_image_standardization(faces)
        face_recognition_input = faces
        embeddings = []
        embeddings = embeddings_extractor(face_recognition_input.to(self.device))
        similarities = torch.tensordot(embeddings, embeddings.T,dims = 1).detach().cpu().numpy()
        # dot product attention without scaling
        # similarities = np.dot(np.array(embeddings), np.array(embeddings).T)
        
        #use the helper function to generate clusters
        components = self._generate_connected_components(
            similarities
        )
        components = [sorted(component) for component in components]

        #assigning each cluster to a unique identity.
        clustered_faces = {}
        for identity_index, component in enumerate(components):
            for index, face_index in enumerate(component):
                component[index] = self.crops[face_index]
            
            clustered_faces[identity_index] = component

        return clustered_faces
    
def cluster(clust : FaceCluster, video_path : str, faces : List[tuple], pad_constant : int | tuple |None = 3) -> Dict[int,list]:
    crops = extract_crops(video_path,faces, pad_constant)
    clustered_faces = clust.cluster_faces(crops)
    return clustered_faces