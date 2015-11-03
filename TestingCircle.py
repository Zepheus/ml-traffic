from cross_validation import cross_validate
from image_loader import load
from learn import LogisticRegressionTrainer
from features import DetectCircle, HogFeature
from visualize import BoxPlot
import numpy as np


def train_folds(directories, trainers):
    images = load(directories, True, permute=True)
    circles = [i for i in images if i.label == "D10"]
    non_circles = [i for i in images if i.label == "B3"]
    # circles = [i for i in images if
    #           i.super_label == "blueCircles" or i.super_label == "red_blue_circles" or i.super_label == "red_circles"]
    # non_circles = [i for i in images if
    #               i.super_label == "squares" or i.super_label == "diamonds" or i.super_label == "reversed_triangles"]
    feature = DetectCircle()

    circle_features = [feature.process(i)[0] for i in circles]
    print("circles done")
    non_circle_features = np.array([feature.process(i)[0] for i in non_circles]).ravel()
    print("non cirlces done")

    BoxPlot().show(["circles", "non_circles"], [circle_features, non_circle_features])


train_folds(["data/train"], LogisticRegressionTrainer(181.0))
