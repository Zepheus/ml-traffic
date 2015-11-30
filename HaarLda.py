from image_loader import load
from features import HaarFeature
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


def lda(directories):
    images = load(directories, True, permute=False)

    f = HaarFeature()
    x = []

    for idx, im in enumerate(images):
        print("%d/%d" % (idx, len(images)))
        x.append(np.array(f.process(im)))

    y_train = [im.label for im in images]
    classes = list(set(y_train))
    class_to_index = {key: index for index, key in enumerate(classes)}
    labels = np.concatenate(np.array([[class_to_index[name] for name in y_train]]))

    clf = ExtraTreesClassifier()
    clf = clf.fit(x, labels)
    w, h = f.size, f.size
    i = 0

    filtered = []
    for size in f.haar_sizes:
        for x in range(w - size):
            for y in range(h - size):
                for haar_type in range(len(f.haars)):
                    score = clf.feature_importances_[i]
                    if score > 0.000001:
                        filtered.append((size, x, y, haar_type, score))
                    i += 1

    sorted_filtered = sorted(filtered, key=lambda tup: tup[4], reverse=True)
    text_file = open("haarImportance.txt", "w")

    for k in sorted_filtered:
        # print("[size=%d][x=%d][y=%d][type=%d] \t=> %f" % k)
        text_file.write("[size=%d][x=%d][y=%d][type=%d] \t=> %f\n" % k)

    text_file.close()


lda(["data/train"])
