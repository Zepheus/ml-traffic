from os.path import basename

# Own packages
from learn import *
from features import *
from Load import *
from image_loader import *
from cross_validation import cross_validate
from skimage.color import *
from visualize import *
from skimage import io
from MetaParameterEstimators import *


def train_and_predict(trainer_function, feature_combiner, number_of_pca_components=0, train_directories=['data/train'],
                      test_directories=['data/test']):
    # Load data
    trainer = trainer_function()
    train_images = load(train_directories, True)
    test_images = load(test_directories, False)
    print('Images loaded')

    # Feature extraction
    feature_extraction(train_images, feature_combiner)
    feature_extraction(test_images, feature_combiner)
    print('Features extracted')

    # Feature transform
    train_data = [image.getFeatureVector() for image in train_images]
    pca = None
    if number_of_pca_components > 0:
        pca = PCA(n_components=min(number_of_pca_components, len(train_images[0].features)))
        pca.fit(train_data)
        train_data = pca.transform(train_data)
    # Train model
    train_classes = [image.label for image in train_images]
    trainer.train(train_data, train_classes)
    print('Training complete!!')

    # Predict class of test images
    file = open('result.csv', 'w')
    file.write('Id,%s\n' % str.join(',', trainer.classes))
    for image in test_images:
        test_data = image.getFeatureVector()
        if pca:
            test_data = pca.transform(test_data)
        predictions = trainer.predict_proba(test_data)
        identifier = int(os.path.splitext(basename(image.filename))[0])
        file.write('%d,%s\n' % (identifier, str.join(',', [('%.13f' % p) for p in predictions[0]])))
    file.close()


def trainFolds(directories, trainers):
    images = load(directories, True, permute=True)
    combiner = [HsvFeature(), DetectCircle(), HogFeature(orientations=5, pixels_per_cell=(8, 8), cells_per_block=(1, 1)),
                 DetectSymmetry(), RegionRatio()]  # Feature selection
    cross_validate(images, combiner, trainers, k=10, use_super_class=False,
                   number_of_pca_components=0, verboseFiles=True)  # use 10 folds, no pca


def estimateMetas(directories):
    meta_estimators = [#estimateColorCenterParameters,
                        estimateHogOrientationsParameters, estimateHogPixelsPerCellParameters,
                        estimateHogCellsPerBlockParameters,
                        estimateDetectCircleParameters
                       ]

    for estimator in meta_estimators:
        estimator(directories, lambda: LogisticRegressionTrainer(181))

#train_and_predict(lambda: LogisticRegressionTrainer(181),
#                  [HsvFeature(), DetectCircle(), HogFeature(), DetectSymmetry(), RegionRatio()], 0,
#                  ['data/train'], ['data/test'])

#test()

#trainFolds(["data/train"], lambda: LogisticRegressionTrainer(181.0))  # Estimated 181 through CV
estimateMetas(['data/train'])
# trainFolds(['data/train/blue_circles','reversed_triangles'])
