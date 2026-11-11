import unittest
from unittest.mock import patch
import numpy as np
from face_detect import detect_faces,IDX_CHIN,IDX_EYE_LEFT,IDX_EYE_RIGHT

class DetectionTileContracts(unittest.TestCase):
    def points(self,x=100):
        p=np.tile(np.array([x,100],np.float32),(468,1))
        p[IDX_CHIN]=[x,140];p[IDX_EYE_LEFT]=[x-10,80];p[IDX_EYE_RIGHT]=[x+10,80]
        return p
    def test_fallback_maps_native_tile_coordinates_into_full_frame(self):
        points=self.points()
        calls=[]
        def detect(image):
            calls.append(image.shape)
            return [points] if len(calls)==5 else []
        with patch('face_detect._detect_landmarks',side_effect=detect):
            faces=detect_faces(np.zeros((800,600,3),np.uint8))
        self.assertEqual(len(faces),1)
        np.testing.assert_array_equal(faces[0].landmarks[IDX_CHIN],[100,290])
        np.testing.assert_array_equal(points[IDX_CHIN],[100,140])
        self.assertEqual(calls[0],(800,600,3));self.assertEqual(calls[1],(300,300,3))
    def test_overlapping_detections_do_not_duplicate_one_face(self):
        p=self.points(250);call=[0]
        def detect(image):
            call[0]+=1
            return [p] if call[0]==2 else ([p-[150,0]] if call[0]==3 else [])
        with patch('face_detect._detect_landmarks',side_effect=detect):
            self.assertEqual(len(detect_faces(np.zeros((800,600,3),np.uint8))),1)
