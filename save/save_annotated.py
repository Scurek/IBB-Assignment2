import os
import cv2
import matplotlib.pyplot as plt


def saveAll(img, prediction_list, path, base_name=""):
    # print(prediction_list)
    for (x, y, w, h) in prediction_list:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # ax = plt.subplot(111)
    # ax.imshow(img)
    # plt.savefig(path + "output_" + base_name)
    cv2.imwrite(os.path.join(path, base_name + ".png"), img)