from identity_cluster.base import detect_faces, extract_crops, _get_crop, _get_frames
import pytest
from typing import Dict, List
import cv2
import os
import numpy as np

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "sample_videos")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

@pytest.fixture
def sample_video_path():
    """Fixture for providing a valid video path."""
    return os.path.join(VIDEOS_DIR, "video1.mp4")

@pytest.fixture
def sample_image_and_bbox():
    """Fixture for providing a valid image and bounding box."""
    image_path = os.path.join(IMAGES_DIR, "image1.jpg")
    bbox = [50, 50, 150, 150]  # Example bounding box
    image = cv2.imread(image_path)
    return image, bbox

def test_detect_faces_valid_video():

    """
    Test function for valid detections (detections with face in it) for a given set of video paths.
    """

    for video_file in os.listdir(VIDEOS_DIR):

        video_path = os.path.join(VIDEOS_DIR, video_file)

        assert os.path.exists(video_path), f"File {video_path} does not exist"

        faces , fps = detect_faces(video_path,"cuda")

        assert isinstance(faces) == Dict[int, List[List[float]]]
        assert isinstance(fps, int)
        assert fps > 0

        for frame_no, boxes in faces.items():
            assert isinstance(frame_no, int), "Frame number should be an integer"
            assert isinstance(boxes, list), "Bounding boxes should be a list"
            for box in boxes:
                assert len(box) == 4, "Each bounding box should have 4 coordinates"

def test_detect_faces_no_faces():
    """
    Test with videos where no faces are present.
    The function should return None for bboxes.
    """
    no_faces_video = os.path.join(VIDEOS_DIR, "no_faces_video.mp4")

    # Ensure video path exists
    assert os.path.exists(no_faces_video), f"Video file {no_faces_video} does not exist"

    # Call the function
    bboxes, fps = detect_faces(no_faces_video, "cpu")

    # Validate output
    assert bboxes is None, "Bounding boxes should be None for videos with no faces"
    assert fps > 0, "FPS should be positive"


def test_get_frames_valid_video(sample_video_path):
    """Test _get_frames with a valid video file."""
    frames = _get_frames(sample_video_path)
    
    # Validate the output
    assert isinstance(frames, list), "Frames should be returned as a list"
    assert len(frames) > 0, "Frames list should not be empty"
    assert all(isinstance(frame, np.ndarray) for frame in frames), "Each frame should be a numpy array"

def test_get_frames_invalid_video():
    """Test _get_frames with an invalid video file."""
    invalid_video_path = "non_existent.mp4"
    with pytest.raises(Exception):
        _get_frames(invalid_video_path)

def test_get_crop_with_int_padding(sample_image_and_bbox):
    """Test _get_crop with integer padding."""
    image, bbox = sample_image_and_bbox
    padding = 10  # Integer padding
    cropped_image = _get_crop(image, bbox, padding)

    # Validate the output
    assert isinstance(cropped_image, np.ndarray), "Cropped image should be a numpy array"
    assert cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0, "Cropped image dimensions should be positive"

def test_get_crop_with_tuple_padding(sample_image_and_bbox):
    """Test _get_crop with tuple padding."""
    image, bbox = sample_image_and_bbox
    padding = (10, 20)  # Tuple padding
    cropped_image = _get_crop(image, bbox, padding)

    # Validate the output
    assert isinstance(cropped_image, np.ndarray), "Cropped image should be a numpy array"
    assert cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0, "Cropped image dimensions should be positive"

def test_get_crop_invalid_padding(sample_image_and_bbox):
    """Test _get_crop with invalid padding."""
    image, bbox = sample_image_and_bbox
    invalid_padding = "invalid"  # Invalid padding type

    with pytest.raises(ValueError):
        _get_crop(image, bbox, invalid_padding)