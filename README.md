# *Installation*

Follow the following instructions for installing the identity_cluster package

1. First, download this bashscript from drive: https://drive.google.com/file/d/1JlJhwA9fJ6kFoEigPFTsaODg4dO_X02D/view?usp=sharing

2. Create and activate a new environment.

3. open git bash in the proper working directory and run the following command

  ```bash identity_cluster_run.sh```


# *Basic Functionalities and Workflows*

The package provides basic functionalities, such as basic video processing and clustering, also the package provides workflows like moving data to s3 and mongodb.

### functionalities:

the main module contains all the basic functionalities involved in video processing and identity management in deepcheck.

1. base
2. cluster

The base submodules support basic video processing methods that are essential for deepcheck, the methods provided are as follows,

---
#### base
---
Here documentation for the classes and functions found in the submodule:

### `VideoFaceDetector` Class (Abstract)
An abstract class for detecting faces in videos.

- **Constructor:**
  ```python
  __init__(self, **kwargs) -> None
  ```
  - Initializes the face detector with additional parameters.

- **Abstract Properties:**
  - `_batch_size`
    - Should return the batch size for face detection.

- **Abstract Methods:**
  - `detect_faces(self, frame)`
    - Should define how faces are detected in a frame.

---


### `FacenetDetector` Class
A concrete implementation of the `VideoFaceDetector` class that uses the `MTCNN` model from the `facenet_pytorch` library for face detection.

- **Constructor:**
  ```python
  __init__(self, **kwargs) -> None
  ```
  - Initializes the detector using the `MTCNN` model and additional parameters.

- **Methods:**
  - `detect_faces(self, frame)`
    - Detects faces in a given video frame using the MTCNN model.
    - **Args:**
      - `frame`: Image frame from the video.
    - **Returns:**
      - List of bounding boxes for detected faces in the frame.

---

### `detect_faces` Method
Detects faces in a video frame using the MTCNN model.

```python
def detect_faces(self, frame) -> list
```
- **Args:**
  - `frame`: Image frame from the video.
- **Returns:**
  - List of bounding boxes (coordinates) of detected faces.


---

### `_get_frames` Function
Retrieves all the frames from a video.

```python
def _get_frames(video_path: str) -> list
```
- **Args:**
  - `video_path`: Path to the video file.
- **Returns:**
  - List of all frames in the video.

---

### `_get_crop` Function
Crops a region from a frame based on a bounding box.

```python
def _get_crop(frame, bbox) -> np.ndarray
```
- **Args:**
  - `frame`: Image frame from the video.
  - `bbox`: Bounding box containing coordinates for cropping.
- **Returns:**
  - Cropped face image.

---

### `extract_crops` Function
Extracts faces from individual frames using bounding boxes.

```python
def extract_crops(video_path, bboxes_dict, get_bboxes=False) -> List[tuple]
```
- **Args:**
  - `video_path`: Path to the video.
  - `bboxes_dict`: Dictionary containing frame numbers and bounding boxes.
- **Returns:**
  - List of tuples with:
    - `frame_no (int)`: Frame number.
    - `PIL.Image`: Cropped face image.
    - `bbox (list)`: Bounding box coordinates.

This provides a detailed documentation of the methods and functions in the file.

---
#### cluster
---

### `FaceCluster` Class
The `FaceCluster` class clusters faces from video frames using a similarity threshold based on dot product similarity between face embeddings. It uses the `InceptionResnetV1` model from the `facenet_pytorch` library for embedding extraction and `networkx` for clustering based on graph connectivity.

- **Constructor:**
  ```python
  def __init__(self, crops=None, similarity_threshold: int = 0.85, device: str = "cpu")
  ```
  - **Args:**
    - `crops`: Optional list of tuples containing `(frame_no (int), PIL Image of cropped face, bbox (list of int))`. Defaults to `None`.
    - `similarity_threshold`: Float value representing the similarity threshold for clustering. Defaults to `0.85`.
    - `device`: String specifying the device for PyTorch computations. Defaults to `"cpu"`.

---

### `_set_crops` Method
Sets the `crops` attribute with the provided list of face crops.

```python
def _set_crops(self, crops)
```
- **Args:**
  - `crops`: List of tuples containing `(frame_no (int), PIL Image of cropped face, bbox (list of int))`.

---

### `_set_threshold` Method
Sets the similarity threshold for clustering.

```python
def _set_threshold(self, threshold)
```
- **Args:**
  - `threshold`: Float value representing the new similarity threshold.

---

### `_preprocess_images` Method
Resizes images to a specified shape.

```python
def _preprocess_images(self, img, shape=[128, 128])
```
- **Args:**
  - `img`: NumPy array representing the image to be resized.
  - `shape`: List of two integers specifying the shape of the output image. Defaults to `[128, 128]`.

- **Returns:**
  - `img`: Resized image as a NumPy array.

---

### `_generate_connected_components` Method
Generates connected components (clusters) based on the similarity matrix.

```python
def _generate_connected_components(self, similarities)
```
- **Args:**
  - `similarities`: NumPy array containing the similarity matrix between face embeddings.

- **Returns:**
  - `components`: List of clusters, where each cluster is a list of indices representing similar faces.

---

### `cluster_faces` Method
Clusters faces based on dot product similarity of face embeddings.

```python
def cluster_faces(self, crops, threshold=None)
```
- **Args:**
  - `crops`: List of tuples containing `(frame_no (int), PIL Image of cropped face, bbox (list of int))`.
  - `threshold`: Optional float value to set or override the similarity threshold.

- **Returns:**
  - `clustered_faces`: Dictionary where keys are cluster identities (integers) and values are lists of face crops associated with each identity.

This covers the `FaceCluster` class and its methods.

---
### deep_check.functionalities.utils
---

This submodule is not intended for using outside the package, but provides the following methods and functionalities

### `detect_probable_fakes` Function
Detects whether a face is likely fake based on the mask and bounding box.

```python
def detect_probable_fakes(mask_frame: np.ndarray, bbox: list, threshold: float = 0.50) -> str
```
- **Args:**
  - `mask_frame`: NumPy array representing the masked frame (image) from which to crop.
  - `bbox`: List containing the bounding box coordinates for cropping the face.
  - `threshold`: Float value representing the probability threshold for determining if the face is fake. Defaults to `0.50`.

- **Returns:**
  - `str`: Returns `"Fake"` if the probability of being fake exceeds the threshold; otherwise, returns `"Real"`.

---

### `cluster` Function
Clusters faces detected in a video using the provided `FaceCluster` instance.

```python
def cluster(clust: FaceCluster, video_path: str, faces: List[tuple]) -> Dict[int, list]
```
- **Args:**
  - `clust`: An instance of the `FaceCluster` class used for clustering faces.
  - `video_path`: String representing the path to the video file.
  - `faces`: List of tuples containing information about detected faces in the video.

- **Returns:**
  - `Dict[int, list]`: Returns a dictionary where keys are cluster identities (integers) and values are lists of face crops associated with each identity.

---

### `get_video_config` Function
Generates a configuration dictionary for a specific identity based on clustered faces.

```python
def get_video_config(clustered_faces: Dict[int, list], identity: int, identity_dir: str | os.PathLike, mask_frames: List[np.ndarray] = None) -> Dict[str, object]
```
- **Args:**
  - `clustered_faces`: Dictionary containing clustered faces, where keys are identity indices and values are lists of face crops and bounding boxes.
  - `identity`: Integer representing the specific identity for which to generate the configuration.
  - `identity_dir`: Path-like object or string representing the directory to save cropped face images and metadata.
  - `mask_frames`: Optional list of NumPy arrays representing the masked frames for each video frame. Defaults to `None`.

- **Returns:**
  - `Dict[str, object]`: Returns a dictionary containing the configuration for the specified identity, including the ID, path, class, and bounding boxes.


---

# Inference Class Documentation

The `Inference` class provides functionalities for inferencing video files with models. It includes utilities for timing and tracking nested function calls to aid in performance analysis.

## Table of Contents
- [Class Attributes](#class-attributes)
- [Methods](#methods)
- [Usage](#usage)
- [License](#license)

---

## Class Attributes
- `device (str)`: Specifies the device (`cpu` or `cuda`) used for computation.
- `shape (tuple)`: Dimensions for resizing clustered faces, default is `(224, 224)`.
- `classes (list)`: The categories the model can predict, such as `["Real", "Fake"]`.
- `timings (dict)`: Stores timing data for functions with nested call relationships.
- `_clusters (None or dict)`: Stores clusters of faces post-clustering.

---

## Methods

### `__init__(self, device: str, shape=(224, 224))`
- Initializes the `Inference` class with specified device and shape attributes.

### `generate_video_data(self, video_path: str, print_timings=True)`
- Processes a video to detect, crop, and cluster faces, and converts them to RGB format.

### `get_data(self, video_path: str, print_timings=True)`
- Retrieves essential data for frames, bounding boxes, images, FPS, and clustered identities from a video.

### `get_predictions(self, model, images: torch.Tensor, device='cuda')`
- Runs predictions on clustered face images using a model and returns logits and labels.

### `__cvt_to_rgb(self, faces: tuple) -> torch.Tensor`
- Converts images from BGR to RGB format.

### `__plot_images_grid(self, tensor: torch.Tensor, images_per_row=4)`
- Plots a grid of images from a 4D tensor.

### `__print_result(self, result: dict, image_data: List[torch.Tensor])`
- Displays results, including images, based on the model’s predictions.

### `__print_timings(self, timings: dict)`
- Outputs timing information for functions with nested call relationships.

### `__create_sequence_dict(self, identity_data)`
- Organizes identity information into a sequence dictionary.

### `__draw_bounding_boxes(self, video_path: str, sequence_dict, result_video_path: str)`
- Draws bounding boxes on faces in the video and saves the processed output.

### Decorator `timeit(func)`
- Times functions and records their nested relationships in the `timings` attribute.

---

## Usage

Here's a simple example of using the `Inference` class:

```python
from inference import Inference
import torch
from models.models_list import ModelList

# Initialize Inference class
device = "cuda" if torch.cuda.is_available() else "cpu"
inference = Inference(device=device)

# Load your model from ModelList
model = ModelList().load_model("face_detection_model", device)

# Process video
video_path = "/path/to/video.mp4"
output_data, num_clusters = inference.generate_video_data(video_path)

# Get predictions for each cluster
for cluster_images in output_data:
    predictions = inference.get_predictions(model, cluster_images, device)
    inference.__print_result(predictions, cluster_images)
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---
