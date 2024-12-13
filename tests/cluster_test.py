import pytest
from unittest.mock import MagicMock
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
from identity_cluster.cluster import cluster, FaceCluster
from identity_cluster.base import extract_crops

@pytest.fixture
def mock_video_path():
    return "C://Users//ASUS//Desktop//Phosphene.AI//identity_cluster//tests//sample_videos//train_00000000.mp4"

@pytest.fixture
def mock_faces():
    # Mock faces with frame_no, PIL image, and bbox
    return {
        0 : [[0,0,128,128]],
        1 : [[10,10,138,138]],
        2 : [[20,20,158,158]]
    }




@pytest.fixture
def mock_face_cluster():
    # Create a mock FaceCluster object
    cluster_instance = FaceCluster(similarity_threshold=0.85, device="cpu")

    # Mock the _preprocess_images function
    cluster_instance._preprocess_images = MagicMock(
        side_effect=lambda img, shape: np.array(img.resize(shape))
    )

    # Mock the embeddings extractor
    cluster_instance._generate_connected_components = MagicMock(
        side_effect=lambda similarities: [[0, 1], [2]]
    )

    # Mock InceptionResnetV1 embeddings
    cluster_instance.cluster_faces = MagicMock(
        side_effect=lambda crops: {
            0: [(0, Image.new('RGB', (128, 128)), [0, 0, 128, 128],Image.new('RGB', (128, 128))),
                (1, Image.new('RGB', (128, 128)), [10, 10, 138, 138],Image.new('RGB', (128, 128)))],
            1: [(2, Image.new('RGB', (128, 128)), [20, 20, 148, 148],Image.new('RGB', (128, 128)))],
        }
    )

    return cluster_instance

@pytest.fixture
def mock_extract_crops():
    # Mock the extract_crops function
    return MagicMock(
        side_effect=lambda video_path, faces, pad_constant: [
            (0, Image.new('RGB', (128, 128)), [0, 0, 128, 128], Image.new('RGB', (128, 128))),
            (1, Image.new('RGB', (128, 128)), [10, 10, 138, 138], Image.new('RGB', (128, 128))),
            (2, Image.new('RGB', (128, 128)), [20, 20, 148, 148], Image.new('RGB', (128, 128))),
        ]
    )

def test_cluster(mock_video_path, mock_faces, mock_face_cluster, mock_extract_crops):
    """
    Test the cluster function to verify correct clustering of face crops.
    """
    
    extract_crops = mock_extract_crops(mock_video_path, mock_faces,3)

    # Call the cluster function
    clustered_faces = cluster(
        clust=mock_face_cluster,
        video_path=mock_video_path,
        faces=mock_faces,
        pad_constant=3
    )

    # Assert the clustering result
    assert isinstance(clustered_faces, dict), "Clustered faces should be a dictionary."
    assert len(clustered_faces) == 2, "There should be 2 clusters."

    # Verify the contents of each cluster
    assert len(clustered_faces[0]) == 2, "First cluster should contain 2 faces."
    assert len(clustered_faces[1]) == 1, "Second cluster should contain 1 face."

    # Assert the mock function calls
    mock_face_cluster.cluster_faces.assert_called()
    mock_extract_crops.assert_called_with(mock_video_path, mock_faces, 3)
