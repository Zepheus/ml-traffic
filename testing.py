from cross_validation import cross_validate
from image_loader import load
from learn import LogisticRegressionTrainer
from features import HsvFeature
from visualize import BoxPlot
import numpy as np
import os


def test_feature(directories, trainers):
    images = load(directories, True, permute=False)
    for image in images:
        if '00085_02840.png' in image.filename:
            feature = HsvFeature()
            feature.process(image)

#     circles = [i for i in images if i.label == "D10"]
#     non_circles = [i for i in images if i.label == "B3"]
#     # circles = [i for i in images if
#     #           i.super_label == "blueCircles" or i.super_label == "red_blue_circles" or i.super_label == "red_circles"]
#     # non_circles = [i for i in images if
#     #               i.super_label == "squares" or i.super_label == "diamonds" or i.super_label == "reversed_triangles"]
#     feature = DetectCircle()
#
#     feature.process(circles[2])
#     # circle_features = [feature.process(i)[0] for i in circles]
#     # print("circles done")
#     # non_circle_features = np.array([feature.process(i)[0] for i in non_circles]).ravel()
#     # print("non cirlces done")
#
#     # BoxPlot().show(["circles", "non_circles"], [circle_features, non_circle_features])


test_feature(["data/train/triangles/A7A"], LogisticRegressionTrainer(181.0))
