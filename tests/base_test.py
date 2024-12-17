from identity_cluster.base import detect_faces, extract_crops, _get_crop, _get_frames
import pytest
from typing import Dict, List
import cv2
import os
import numpy as np
from PIL import Image

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "sample_videos")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "sample_images")

@pytest.fixture
def sample_video_path():
    """Fixture for providing a valid video path."""
    return os.path.join(VIDEOS_DIR, "train_00000000.mp4")

@pytest.fixture
def sample_image_and_bbox():
    """Fixture for providing a valid image and bounding box."""
    image_path = os.path.join(IMAGES_DIR, "image1.jpeg")
    bbox = [50, 50, 150, 150]  # Example bounding box
    image = cv2.imread(image_path)
    return image, bbox

@pytest.fixture
def sample_bboxes():
    """Fixture for providing sample bounding boxes for testing."""
    # Synthetic bounding boxes for a video with at least 2 frames
    return {
        0: [[50, 50, 200, 200]],  # Frame 0 has one face
        1: [[30, 40, 100, 120], [150, 150, 300, 300]],  # Frame 1 has two faces
    }

@pytest.fixture
def sample_flow_videos():
    """Fixture for providing 7 videos for testing"""
    return [
        os.path.join(VIDEOS_DIR,"simran.mp4"),
        os.path.join(VIDEOS_DIR,"test_00000000.mp4"),
        os.path.join(VIDEOS_DIR,"test_00000011.mp4"),
        os.path.join(VIDEOS_DIR,"test_00000131.mp4"),
        os.path.join(VIDEOS_DIR,"test_00000121.mp4"),
        os.path.join(VIDEOS_DIR,"test_00000214.mp4"),
        os.path.join(VIDEOS_DIR,"test_00001395.mp4")
                ]   
def test_detect_faces_valid_video(sample_flow_videos):

    """
    Test function for valid detections (detections with face in it) for a given set of video paths.
    """

    for video_path in sample_flow_videos:

        assert os.path.exists(video_path), f"File {video_path} does not exist"

        faces , fps = detect_faces(video_path,"cuda")
        assert isinstance(faces, dict) , "faces must be dictionary object with key being frame numbers and values being list of bounding boxes"
        assert isinstance(fps, int) , "fps must be a valid number above 0"
        assert fps > 0

        for frame_no, boxes in faces.items():
            if not boxes:
                continue
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

def test_extract_crops_valid_input(sample_video_path, sample_bboxes):
    """Test extract_crops with valid input."""
    pad_constant = 20
    crops = extract_crops(sample_video_path, sample_bboxes, pad_constant)

    # Validate output
    assert isinstance(crops, list), "Output should be a list of crops"
    assert len(crops) > 0, "Crops list should not be empty"

    for crop in crops:
        frame_no, cropped_face, bbox, padded_crop = crop
        assert isinstance(frame_no, int), "Frame number should be an integer"
        assert isinstance(cropped_face, Image.Image), "Cropped face should be a PIL Image"
        assert isinstance(bbox, list) and len(bbox) == 4, "Bounding box should be a list of 4 integers"
        assert isinstance(padded_crop, Image.Image), "Original crop should be a PIL Image"

def test_extract_crops_no_bboxes(sample_video_path):
    """Test extract_crops with no bounding boxes."""
    empty_bboxes = {}  # No bounding boxes
    crops = extract_crops(sample_video_path, empty_bboxes)

    # Validate output
    assert isinstance(crops, list), "Output should be a list"
    assert len(crops) == 0, "Crops list should be empty when there are no bounding boxes"


def test_extract_crops_large_padding(sample_video_path, sample_bboxes):
    """Test extract_crops with large padding."""
    pad_constant = 200  # Large padding
    crops = extract_crops(sample_video_path, sample_bboxes, pad_constant)

    # Validate output
    assert isinstance(crops, list), "Output should be a list of crops"
    assert len(crops) > 0, "Crops list should not be empty"

    for crop in crops:
        frame_no, cropped_face, bbox, padded_crop = crop
        assert isinstance(cropped_face, Image.Image), "Cropped face should be a PIL Image"
        assert cropped_face.size[0] > 0 and cropped_face.size[1] > 0, "Cropped face dimensions should be positive"

def test_flow(sample_flow_videos):
    """
    Test the detect_faces and extract_crops phases.
    
    Args:
        sample_flow_videos (List[str]): List of paths to sample video files for testing.
    """
    for video_path in sample_flow_videos:
        # Detect faces
        faces, fps = detect_faces(video_path, "cpu")

        # Validate faces
        assert isinstance(faces, dict), f"Expected faces to be a dict, got {type(faces)}"
        for frame_no, bboxes in faces.items():
            if not bboxes:
                continue
            assert isinstance(frame_no, int), f"Expected frame_no to be int, got {type(frame_no)}"
            assert isinstance(bboxes, list), f"Expected bboxes to be a list, got {type(bboxes)}"
            for bbox in bboxes:
                assert isinstance(bbox, list), f"Expected bbox to be a list, got {type(bbox)}"
                assert len(bbox) == 4, f"Expected bbox to have 4 elements, got {len(bbox)}"
                assert all(isinstance(coord, (int, float)) for coord in bbox), \
                    f"Expected bbox coordinates to be int or float, got {bbox}"

        # Validate fps
        assert isinstance(fps, int), f"Expected fps to be int, got {type(fps)}"
        assert fps > 0, "FPS should be greater than 0"

        # Extract crops
        crops = extract_crops(video_path, faces)
        assert isinstance(crops, list), f"Expected crops to be a list, got {type(crops)}"
        assert len(crops) > 0, "Crops list should not be empty"

        for crop in crops:
            assert isinstance(crop, tuple), f"Expected crop to be a tuple, got {type(crop)}"
            assert len(crop) == 4, f"Expected crop tuple to have 4 elements, got {len(crop)}"
            frame_no, cropped_face, bbox, padded_crop = crop

            # Validate frame number
            assert isinstance(frame_no, int), f"Expected frame_no to be int, got {type(frame_no)}"

            # Validate cropped_face
            assert isinstance(cropped_face, Image.Image), \
                f"Expected cropped_face to be PIL Image, got {type(cropped_face)}"

            # Validate bbox
            assert isinstance(bbox, list), f"Expected bbox to be list, got {type(bbox)}"
            assert len(bbox) == 4, f"Expected bbox to have 4 elements, got {len(bbox)}"
            assert all(isinstance(coord, (int, float)) for coord in bbox), \
                f"Expected bbox coordinates to be int or float, got {bbox}"

            # Validate padded_crop
            assert isinstance(padded_crop, Image.Image), \
                f"Expected padded_crop to be PIL Image, got {type(padded_crop)}"




