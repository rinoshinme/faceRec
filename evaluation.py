"""
RUNNING TIME
ACCURACY
"""

from face_recognizer import FaceRecognizer
import time
import os
import numpy as np


def time_fr():
    embedding_file = 'embeddings.npy'
    if os.path.exists(embedding_file):
        os.remove(embedding_file)

    face_recog = FaceRecognizer(model_path='model/20180402-114759')
    image_path = 'faces.jpg'
    image = face_recog.read_image_sp(image_path)

    det_start = time.time()
    bboxes, scores, keypoints = face_recog.detect_faces_and_keypoints(image)
    det_stop = time.time()
    print('detection time = ', det_stop - det_start)

    rec_start = time.time()
    face_patches = face_recog.prepare_image_patches(image, bboxes, 160, 20)
    embeddings = face_recog.extract_embedding(face_patches)
    rec_stop = time.time()
    print('recognition time = ', rec_stop - rec_start)

    np.save(embedding_file, embeddings)


if __name__ == '__main__':
    time_fr()






