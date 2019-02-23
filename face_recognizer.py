from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mtcnn import detect_face
import facenet
from scipy import misc
import tensorflow as tf
import os
import numpy as np
import cv2


class FaceRecognizer(object):
    def __init__(self, model_path):
        with tf.Graph().as_default():
            # mtcnn model
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self.sess_mtcnn = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.sess_mtcnn.as_default():
                # nets = [pnet, rnet, onet]
                self.nets = detect_face.create_mtcnn(self.sess_mtcnn, None)

            # facenet model
            self.sess_facenet = tf.Session()
            with self.sess_facenet.as_default():
                facenet.load_model(model_path)
                self.image_tensor = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embedding_tensor = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                # True for training and False for testing
                self.phase_tensor = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print('load model success')

        # define parameters
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

    def cv_scipy(self, image):
        image = image[:, :, ::-1]
        return image.copy()

    def read_image_sp(self, image_path):
        img = misc.imread(image_path, mode='RGB')
        return img

    def read_image_cv(self, image_path):
        img = cv2.imread(image_path)
        return img

    def detect_faces_and_keypoints(self, image):
        """
        Detect faces from a image, return face bounding boxes and face key points.
        face boxes maybe further filtered with scores and keypoints
        """
        bounding_boxes, markers = detect_face.detect_face(image, self.minsize, self.nets[0], self.nets[1],
                                                          self.nets[2], self.threshold, self.factor)

        num_faces = bounding_boxes.shape[0]
        bboxes = []
        scores = []
        keypoints = []
        for i in range(num_faces):
            bboxes.append(bounding_boxes[i, 0:4])
            scores.append(bounding_boxes[i, 4])
            pts = []
            for k in range(5):
                pts.append((int(markers[k ,i]), int(markers[5 + k, i])))
            keypoints.append(pts)

        return bboxes, scores, keypoints

    def prepare_image_patches(self, image, bboxes, image_size, margin):
        """
        crop image patches, do resizing and prewhitening
        """
        img_size = np.asarray(image.shape)[0:2]
        n_faces = len(bboxes)
        faces = []
        for n in range(n_faces):
            det = bboxes[n]
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            faces.append(prewhitened)
        faces = np.stack(faces)
        return faces

    def extract_embedding(self, images):
        """
        extract face embedding from prewhitened image patch.
        """
        feed_dict = {self.image_tensor:images, self.phase_tensor:False}
        embeddings = self.sess_facenet.run(self.embedding_tensor, feed_dict=feed_dict)
        return embeddings

    def similarity(self, embed1, embed2):
        diff = embed1 - embed2
        length = np.sum(diff * diff)
        return 1.0 - length

    def draw_detection(self, image, bboxes, keypoints, draw_bbox=True, draw_keypoints=True):
        """
        draw bbox and keypoints on cv2 image
        """
        n_faces = len(bboxes)
        for n in range(n_faces):
            if draw_bbox:
                tl = (int(bboxes[n][0]), int(bboxes[n][1]))
                br = (int(bboxes[n][2]), int(bboxes[n][3]))
                cv2.rectangle(image, tl, br, (255, 0, 0), thickness=2)
            if draw_keypoints:
                for pt in keypoints[n]:
                    cv2.drawMarker(image, pt, (0, 0, 255), cv2.MARKER_DIAMOND)
        return image

    def show_results(self, image, bboxes, keypoints):
        cv_image = self.cv_scipy(image)
        cv_image = self.draw_detection(cv_image, bboxes, keypoints)
        cv2.imshow('image', cv_image)
        cv2.waitKey(0)


def embedding_similarity(embed1, embed2):
    diff = embed1 - embed2
    length = np.sum(diff * diff)
    return 1.0 - length


def main():
    embedding_file = 'embeddings.npy'
    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file)
    else:
        face_recog = FaceRecognizer(model_path='model/20180402-114759')
        image_path = 'faces.jpg'
        image = face_recog.read_image_sp(image_path)
        bboxes, scores, keypoints = face_recog.detect_faces_and_keypoints(image)
        # face_recog.show_results(image, bboxes, keypoints)

        face_patches = face_recog.prepare_image_patches(image, bboxes, 160, 20)
        embeddings = face_recog.extract_embedding(face_patches)
        np.save(embedding_file, embeddings)

    for i in range(25):
        print(embedding_similarity(embeddings[0, :], embeddings[i, :]))


'''
facenet as a feature extractor, construct another mlp model to classify different faces.
face CLUSTERING using [algorithm?] appeared in dlib face recognition example
'''
if __name__ == '__main__':
    main()
