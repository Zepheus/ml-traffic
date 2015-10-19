from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA

# own packages
from image_loader import *


def cross_validate(images, feature_combiner, trainer_function, k=1, use_super_class=True, number_of_pca_components=0):
    kf = KFold(len(images), n_folds=k)
    i = 1
    error_ratios = []
    for trainIndices, testIndices in kf:
        print('-------- calculating fold %d --------' % i)
        # Split in folds
        train_images = [images[i] for i in trainIndices]
        test_images = [images[i] for i in testIndices]
        # Feature extraction
        feature_extraction(train_images, feature_combiner)
        feature_extraction(test_images, feature_combiner)
        # Train
        trainer = trainer_function()
        train_data = [image.features for image in train_images]
        pca = None
        if number_of_pca_components > 0:
            pca = PCA(n_components=min(number_of_pca_components, len(train_images[0].features)))
            pca.fit(train_data)
            train_data = pca.transform(train_data)
        train_classes = [image.super_label if use_super_class else image.label for image in train_images]
        trainer.train(train_data, train_classes)
        # Predict
        test_data = [image.features for image in test_images]
        if pca is not None:
            test_data = pca.transform(test_data)  # perform PCA
        predictions = trainer.predict(test_data)
        # Compare predictions with ground truths
        ground_truths = [image.super_label if use_super_class else image.label for image in test_images]
        errors = []
        for image, prediction, ground_truth in zip(test_images, predictions, ground_truths):
            if prediction != ground_truth:
                print('\r    [ERROR] for image %s I predicted "%s" but the sign actually was "%s"' % (
                    image.filename, prediction, ground_truth))
                errors.append([image, prediction, ground_truth])
        sys.stdout.write('\r    test calculation [100 %]\n')
        error = float(len(errors)) / len(test_images)
        print('    error ratio of fold: %f' % error)
        error_ratios.append(error)
        i += 1
    print('-------- folds done --------\n')
    print('average errorRatio is %f' % np.mean(error_ratios))
