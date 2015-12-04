from image_loader import load
from learn import LogisticRegressionTrainer
from features import ColorFeature


def test_feature(directories, trainers):
    images = load(directories, True, permute=False)
    for image in images:
        if '01005_05517' in image.filename:
            feature = ColorFeature()
            feature.process(image)

#     circles = [i for i in images if i.label == "D10"]
#     non_circles = [i for i in images if i.label == "B3"]
#     # circles = [i for i in images if
#     #           i.super_label == "blueCircles" or i.super_label == "red_blue_circles" or i.super_lab  el == "red_circles"]
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


test_feature(["data/train/blue_circles/D1a"], LogisticRegressionTrainer(181.0))
