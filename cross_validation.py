from sklearn.cross_validation import KFold
import itertools
import random

# own packages
from image_loader import *
from features import *
from preps import ResizeTransform
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


def split_kfold(images, k):
    kf = KFold(len(images), n_folds=k)
    return [([images[i] for i in trainIndices], [images[i] for i in testIndices]) for trainIndices, testIndices in kf]


def extract_pole_nr(image):
    _, tail = os.path.split(image.filename)
    return int(tail.split('_')[0])


def split_special(images, maxFolds=10, limit=False):
    class_by_pole = list([list([list(image_by_pole)
                                for _, image_by_pole in
                                itertools.groupby(sorted(img_by_class, key=lambda l: extract_pole_nr(l)),
                                                  lambda y: extract_pole_nr(y))])
                          for cl, img_by_class in
                          itertools.groupby(sorted(images, key=lambda p: p.label), lambda x: x.label)])
    if limit:
        max_allowed_folds = min(maxFolds, min([len(x) for x in class_by_pole]))
    else:
        max_allowed_folds = maxFolds

    # Now we make sure that for each class, it is added to at least each fold
    # Warning: folds = [[]] * max_allowed_folds uses the same array reference for each!
    folds = []
    for fold in range(max_allowed_folds):
        folds.append([])

    for cl in class_by_pole:
        permutations = list(range(len(cl)))
        random.shuffle(permutations)
        if limit:
            # Spread permutation evenly over all folds
            while len(permutations) >= max_allowed_folds:
                for fold in range(max_allowed_folds):  # spread over the available folds
                    image_group = cl[permutations.pop(0)]
                    folds[fold].extend(image_group)

        # Now take care of the remaining items (distribute randomly, no other choice)
        for i in range(len(permutations)):
            fold = random.randrange(len(folds))
            image_group = cl[permutations.pop(0)]
            folds[fold].extend(image_group)

    # Now that we generated the folds, we generate fold sets
    return [(list(itertools.chain(*[folds[k] for k in [j for j in range(max_allowed_folds) if j != i]])), folds[i]) for
            i in range(max_allowed_folds)]


def single_validate(trainer, train_data, train_classes, test_data, test_classes, test_images, verbose, verboseFiles):
    # Train
    trainer.train(train_data, train_classes)

    # Predict
    predictions = trainer.predict(test_data)

    # Compare predictions with ground truths
    errors = []
    for image, prediction, ground_truth in zip(test_images, predictions, test_classes):
        if prediction != ground_truth:
            errors.append([image, prediction, ground_truth])
            if verbose and verboseFiles:
                print('\r    [ERROR] for image %s I predicted "%s" but the sign actually was "%s"' % (
                    image.filename, prediction, ground_truth))
    if verbose:
        sys.stdout.write('\r    test calculation [100 %]\n')
    error = float(len(errors)) / len(test_classes)
    return error


def extract_resized_images(images, size):
    print('Extracting raw resized images at size %d' % size)
    num_images = len(images)
    transform = ResizeTransform(size)
    buf = np.zeros((num_images, size, size, 3))
    for i, img in enumerate(images):
        img_data = transform.process(img.image)
        buf[i, :, :, :] = img_data
    return buf


def cross_validate(images, feature_combiner, trainer_function, k=10, augmented=True,
                   verbose=True, verboseFiles=False):
    # fold = split_kfold(images, k)

    fold = split_special(images, k)
    if verbose:
        print('Split into %d folds' % len(fold))

    multitrain = isinstance(trainer_function, list)
    error_ratios = []

    if multitrain:
        for i in range(len(trainer_function)):
            error_ratios.append([])

    noFeatures = isinstance(feature_combiner, int)

    for i, (train_images, test_images) in enumerate(fold):
        assert len(train_images) + len(test_images) == len(images)
        if verbose:
            print('-------- calculating fold %d --------' % (i + 1))

        # Augment train
        if augmented:
            transforms = list([RotateTransform(degrees) for degrees in [-10, -7.0, 7.0, 10]]) + \
               [SqueezeTransform(), MirrorTransform()]
            train_images = augment_images(train_images, transforms)
            if verbose:
                print('Augmented train images to %d samples' % len(train_images))
        else:
            if verbose:
                print('Traning on %d images.' % len(train_images))
        # Feature extraction
        train_classes = [image.label for image in train_images]
        test_classes = [image.label for image in test_images]
        if not noFeatures:
            feature_extraction(train_images, feature_combiner, verbose=verbose)
            feature_extraction(test_images, feature_combiner, verbose=verbose)

            train_data = [image.get_feature_vector() for image in train_images]
            test_data = [image.get_feature_vector() for image in test_images]
        else:
            train_data = extract_resized_images(train_images, feature_combiner)
            test_data = extract_resized_images(test_images, feature_combiner)

        # Train
        if multitrain:
            for trainer_idx, trainer_factory in enumerate(trainer_function):
                trainer = trainer_factory()
                if verbose:
                    print('    Starting calculating with %s' % str(trainer))
                error = single_validate(trainer, train_data, train_classes, test_data, test_classes, test_images,
                                        verbose, verboseFiles)
                error_ratios[trainer_idx].append(error)
                if verbose:
                    print('    error ratio of fold: %f (trainer %s)' % (error, str(trainer)))
        else:
            trainer = trainer_function()
            if verbose:
                print('    Starting calculating with %s' % str(trainer))
            error = single_validate(trainer, train_data, train_classes, test_data, test_classes, test_images, verbose,
                                    verboseFiles)

            if verbose:
                print('    error ratio of fold: %f' % error)
            error_ratios.append(error)

    if verbose:
        print('-------- folds done --------\n')

    if not multitrain:
        mean_result = np.mean(error_ratios)
        std_result = np.std(error_ratios)
        if verbose:
            print('mean error_ratio is %f (std: %f)' % (mean_result, std_result))
        return mean_result
    else:
        mean_errors = [np.mean(x) for x in error_ratios]
        std_errrors = [np.std(x) for x in error_ratios]
        if verbose:
            for idx, trainer in enumerate(trainer_function):
                trainername = str(trainer())
                print('mean error_ratio %f (std: %f) for %s' % (mean_errors[idx], std_errrors[idx], trainername))
        return mean_errors


def cross_grid_search(directories, trainer, features, parameters, augment=True, verbose=True, numjobs=1):
    print('Grid search using %d jobs' % numjobs)
    images = load(directories, True, permute=False)

    # Augment train
    if augment:
        transforms = list([RotateTransform(degrees) for degrees in [-10, -7.0, 7.0, 10]]) + \
               [SqueezeTransform(), MirrorTransform()]
        images = augment_images(images, transforms)
        print('Augmented train images to %d samples' % len(images))

    feature_extraction(images, features, verbose=verbose)
    train_data = [image.get_feature_vector() for image in images]
    train_classes = [image.label for image in images]
    classes = list(set(train_classes))
    class_to_index = {key: index for index, key in enumerate(classes)}
    labels = np.concatenate(np.array([[class_to_index[name] for name in train_classes]]))

    # TODO: custom CV object (like above)
    for score in ['precision', 'recall']:
        print('Optimizing %s, stay tight...' % score)
        clf = GridSearchCV(trainer, parameters, cv=5, scoring='%s_weighted' % score, n_jobs=numjobs)
        clf.fit(train_data, labels)
        print("Best parameters set found on development set (%s):" % score)
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()


def trainFolds(directories, trainers, features, augment=True, folds=10):
    images = load(directories, True, permute=False)
    cross_validate(images, features, trainers, k=folds, verbose=True, verboseFiles=False, augmented=augment)  # use 10 folds, no pca
