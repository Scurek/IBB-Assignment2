import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
from save.save_annotated import saveAll
from tqdm import tqdm


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        eval_config = config['evaluation']
        self.images_path = eval_config['images_path']
        self.annotations_path = eval_config['annotations_path']
        self.output_path = eval_config['output_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self):
        LIMIT = -1
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        import detectors.cascade_detector.detectorFace as face_detector
        # import detectors.your_super_detector.detector as super_detector

        detectors = {
            "cascade": {
                "detector": cascade_detector.Detector(1.05, 1),
                "predictions": [],
                "iou_arr": [],
                "types": [0, 0, 0]
            },
            # "cascadeFace": {
            #     "detector": face_detector.Detector(1.05, 1),
            #     "predictions": [],
            #     "iou_arr": [],
            #     "types": [0, 0, 0]
            # }
        }

        # for i in tqdm(range(len(im_list)), desc="Evaluating..."):
        for i in range(len(im_list)):
            print(i)
            if 0 < LIMIT < i:
                break
            im_name = im_list[i]
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            # img = preprocess.grayscale(img)
            # img = preprocess.automatic_brightness_and_contrast(img)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = np.array(self.get_annotations(annot_name))
            #
            # saveAll(img, annot_list, self.output_path, "annotated" + "_" + str(i))
            # continue

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            for name, detector in detectors.items():
                prediction_list = detector["detector"].detect(img)
                saveAll(img, prediction_list, self.output_path, name + "_" + str(i))
                p, gt = eval.prepare_for_detection(prediction_list, annot_list)
                iou, class_type = eval.iou_compute(p, gt, 0.5)
                detector["iou_arr"].append(iou)
                detector["types"][class_type] += 1


        for name, detector in detectors.items():
            miou = np.average(detector["iou_arr"])
            print("\n")
            print("(" + name + ") Average IOU:", f"{miou:.2%}")
            print("(" + name + ") T:" + str(detector["types"]))
            print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
