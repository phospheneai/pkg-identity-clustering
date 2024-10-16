# *Installation*

Follow the following instructions for installing the identity_cluster package

1. First, download this bashscript from drive: https://drive.google.com/file/d/1Or1ciQ4MeP4W2pnQZyxTmYzdXasKSm9H/view?usp=sharing

2. Create and activate a new environment.

3. open git bash in the proper working directory and run the following command

  ```bash identity_cluster_run.sh```


# *Basic Functionalities and Workflows*

The package provides basic functionalities, such as basic video processing and clustering, also the package provides workflows like moving data to s3 and mongodb.

### deep_check.functionalities:

this module contains all the basic functionalities involved in video processing and identity management in deepcheck.

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

