import glob
import os
import cv2

class Trainer():
    model = cv2.face.LBPHFaceRecognizer_create()

    def train(self, imgs, labels):
        self.model.train(imgs, labels)

    def save(self, path):
        self.model.write(path)