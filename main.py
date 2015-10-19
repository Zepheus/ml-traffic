from learn import *

# Own packages
from learn import *
from features import *
from Load import *
from image_loader import *
from cross_validation import cross_validate

def train_and_predict(trainer_function, feature_combiner, number_of_pca_components=0, train_directories=['data/train'], test_directories=['data/small']):
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
    train_data = [image.features for image in train_images]
    if number_of_pca_components > 0:
        pca = PCA(n_components=min(number_of_pca_components, len(train_images[0].features)))
        pca.fit(train_data)
        train_data = pca.transform(train_data)
    # Train model
    train_classes = [image.label for image in train_images]
    trainer.train(train_data, train_classes)
    print('Training complete!!')
    # Predict class of test images
    print(trainer.predict_proba(test_images[0].features))



def trainFolds(directories):
    images = load(directories, True, permute=True)
    combiner = FeatureCombiner([HsvFeature(), DetectCircle(), HogFeature()])  # Feature selection
    trainer = GaussianNaiveBayes  # Learning algorithm, make sure this is a function and not an object
    cross_validate(images, combiner, trainer, 3, False, 10)  # use 10 PCA components


#trainer_function = GaussianNaiveBayes
#train_and_predict(trainer_function, FeatureCombiner([HsvFeature(), DetectCircle()]), 10, ['data/train/blue_circles/D10', 'data/train/blue_circles/D9'], ['data/train/blue_circles/D9'])

trainFolds(['data/train/blue_circles/D10', 'data/train/blue_circles/D9'])