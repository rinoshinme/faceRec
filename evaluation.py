"""
RUNNING TIME
ACCURACY
"""

from face_recognizer import FaceRecognizer
import time
import os
import numpy as np
from scipy import misc


def resize_image(image, low=1000, high=1500):
    h, w, _ = image.shape
    short = min(h, w)
    long = max(h, w)
    ratio = low * 1.0 / short
    if long * ratio > high:
        ratio = high * 1.0 / long
    rh = int(h * ratio)
    rw = int(w * ratio)
    return misc.imresize(image, (rh, rw), interp='bilinear')


def time_fr(image_folder=None):
    embedding_file = 'embeddings.npy'
    if os.path.exists(embedding_file):
        os.remove(embedding_file)

    face_recog = FaceRecognizer(model_path='model/20180402-114759')
    if image_folder is None:
        image_path = 'faces.jpg'
        image_paths = [image_path for i in range(10)]
    else:
        image_paths = os.listdir(image_folder)
        image_paths = [os.path.join(image_folder, p) for p in image_paths if p.endswith('jpg')]

    for i, image_path in enumerate(image_paths):
        print('running: %d'%i)
        image = face_recog.read_image_sp(image_path)
        image = resize_image(image)
        det_start = time.time()
        bboxes, scores, keypoints = face_recog.detect_faces_and_keypoints(image)
        det_stop = time.time()
        print('detection time = ', det_stop - det_start)

        if len(bboxes) == 0:
            print('no face detected')
        else:
            rec_start = time.time()
            face_patches = face_recog.prepare_image_patches(image, bboxes, 160, 20)
            embeddings = face_recog.extract_embedding(face_patches)
            rec_stop = time.time()
            print('recognition time = ', rec_stop - rec_start)


if __name__ == '__main__':
    time_fr()






