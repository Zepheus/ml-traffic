# Own
from image_loader import load, augment_images
from cross_validation import split_special
from preps import RotateTransform, SqueezeTransform, MirrorTransform, PerspectiveTransform
# Skimage
from skimage.transform import resize
from skimage import exposure, img_as_float
from skimage.color import rgb2gray
# OS
import time
import gc
# Scientific
import numpy as np
import theano
import theano.tensor as T
# Neural
import lasagne


# Convert images to one numpy array of dimension (images, color_channels, w, h)
def images_to_vectors(imgs, size, num_dimensions=3):
    vect = np.zeros((len(imgs), num_dimensions, size, size), dtype=np.float32)  # Assume 3 channels
    for idx, img in enumerate(imgs):
        if num_dimensions > 1:
            rolled = np.rollaxis(img.getByName('color_%d' % size).astype(np.float32), 2, 0)
            vect[idx, :, :, :] = rolled
        else:
            vect[idx, 0, :, :] = img.getByName('gray_%d' % size).astype(np.float32)
    return vect


def load_images(directories, is_train=False, permute=True):
    return load(directories, is_train, permute)


def kaggle_logloss(act, pred):
    pred = np.clip(pred, 1e-10, 1 - 1e-10)
    return -np.sum(act * np.log(pred)) / act.shape[0]


def logloss(predictions, weights, y_val, num_classes=81):
    assert (abs(sum(weights) - 1) < 0.0001)
    assert(len(predictions) == len(weights))
    num_samples = y_val.shape[0]

    if len(weights) > 1:
        final_pred = np.zeros((num_samples, num_classes))
        for pred, w in zip(predictions, weights):
            final_pred = final_pred + (w * pred)
    else:
        final_pred = predictions[0]

    # Convert to kaggle format
    vals = np.zeros((num_samples, num_classes))
    for idx, val_y in enumerate(y_val):
        vals[idx][val_y] = 1
    return kaggle_logloss(vals, final_pred)


# Resize and/or convert images to grayscale
def postprocess(imgs, size, grayscale=False):
    print("Postprocessing images and resize (at %d)" % size)
    keyname = ('gray_%d' if grayscale else 'color_%d') % size
    for img in imgs:

        # Continue if already calculated
        if img.isSetByName(keyname):
            continue

        floatimg = img_as_float(img.image)
        floatimg = resize(floatimg, (size, size))
        if grayscale:
            floatimg = rgb2gray(floatimg)
        img.setByName(keyname, floatimg)  # expect to return floats


# Augment images
def augmentation(images):
    transforms = list([RotateTransform(degrees) for degrees in [-10, -7.0, 7.0, 10]]) + \
                 [SqueezeTransform(), MirrorTransform()]
    return augment_images(images, transforms)


def print_network(network):
    print('Network:')
    layers = lasagne.layers.get_all_layers(network)
    for layer in layers:
        shape = lasagne.layers.get_output_shape(layer)
        print('\t%s layer with output shape %s' % (type(layer).__name__, shape))


# First neural network structure
def build_rgb_cnn(input_size, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, input_size, input_size), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3, 1),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Maxpooled layer:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.FeaturePoolLayer(network, pool_size=2)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=150,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=81,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


# Build second simple neural network
def build_rgb_cnn_2(input_size, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, input_size, input_size), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=100, filter_size=(10, 10),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=100,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=81,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


# Build grayscale neural network
def build_grayscale_cnn(input_size, input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 1, input_size, input_size), input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=128, filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Maxpooled
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.FeaturePoolLayer(network, pool_size=2)

    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.5),
        num_units=100,
        nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
        network,
        num_units=81,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_network(network, input_var, target_var, learning_rate=0.005, momentum=0.9):
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=learning_rate, momentum=momentum)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # Also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)

    return train_fn, val_fn, predict_fn


# Train the model by iterating over the batches and doing backprop
def train(train_fn, val_fn, pred_fn, X_train, y_train, X_val, y_val, num_epochs=500, show_validation=True):
    print("Starting training...")

    training_loss = 0
    val_loss = 0
    validation_acc = 0
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        if show_validation:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

        timediff = time.time() - start_time
        training_loss = train_err / train_batches
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, timediff))
        print("  training loss:\t\t{:.6f}".format(training_loss))

        if show_validation:
            val_loss = val_err / val_batches
            validation_acc = val_acc / val_batches * 100
            print("  validation loss:\t\t{:.6f}".format(val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(validation_acc))

    print('Finished %d iterations' % num_epochs)

    if show_validation:
        return training_loss, val_loss, validation_acc
    else:
        return training_loss


# Predict probabilities
def predict(predict_fn, x_test):
    num_samples = x_test.shape[0]
    prediction = np.empty((num_samples, 81), dtype=np.float32)
    for i in range(num_samples):  # Test each sample separately due to memory constraints on GPU
        prediction[i, :] = predict_fn(np.array([x_test[i, :, :, :]]))
    return prediction


# Dump probabilities to CSV
def write_csv(test_images, predictions, id_to_class, filename='result.csv', combined=False):
    assert (isinstance(test_images, list))  # do NOT pass a dictionary here

    file = open(filename, 'w')
    file.write('Id,%s\n' % str.join(',', id_to_class))
    for idx, img in enumerate(test_images):
        probs = predictions[idx]
        thesum = np.sum(probs)  # Check if total sum of the array is 1, disregarding some floating point errors
        if abs(thesum - 1.0) > 0.01:
            print('Warning: Incorrect probabilities for %d' % img.identifier)
        if combined:
            # Take care of floating point rounding errors
            file.write(
                '%d,%s\n' % (img.identifier, str.join(',', [('%.13f' % (p if p < 1.0 else 1.0)) for p in probs])))
        else:
            file.write('%d,%s\n' % (img.identifier, str.join(',', [('%.13f' % p) for p in probs])))

    file.close()


# Perform cross validation on a training set
def cross_validate(train_dir, networks, weights, epochs, input_sizes, learning_rates, grays, num_folds=5, augment=True):
    print('Cross-validation using %d folds' % num_folds)
    if isinstance(weights, tuple) and len(weights) > 1:
        print('Using weights %s' % str.join(',', [str(x) for x in weights]))

    train_images = load_images(train_dir, is_train=True, permute=False)
    training_labels = list([img.label for img in train_images])
    classes_set = list(sorted(set(training_labels)))
    class_to_index = {key: index for index, key in enumerate(classes_set)}

    if not isinstance(networks, tuple):
        networks = (networks)
    if not isinstance(weights, tuple):
        weights = (weights)
    if not isinstance(epochs, tuple):
        epochs = (epochs)
    if not isinstance(input_sizes, tuple):
        input_sizes = (input_sizes)
    if not isinstance(learning_rates, tuple):
        learning_rates = (learning_rates)
    if not isinstance(grays, tuple):
        grays = (grays)

    # Create validation and training set
    val_losses = []
    train_losses = []
    val_accs = []

    for trainset, valset in split_special(train_images, num_folds, num_folds == 2):

        # For each network in ensemble (if available)
        predictions = []
        print('Evaluating fold... (%d models in ensemble)' % len(weights))
        if augment:
            trainset = augmentation(trainset)
            print("Augmented to %d images" % len(trainset))

        # Convert classes to uin8 indices
        y_train = np.concatenate(np.array([[class_to_index[img.label] for img in trainset]], dtype=np.uint8))
        y_val = np.concatenate(np.array([[class_to_index[img.label] for img in valset]], dtype=np.uint8))

        for input_size, network, num_epochs, gray, learning_rate in zip(input_sizes, networks, epochs,
                                                                    grays, learning_rates):
            # Postprocess images
            postprocess(trainset, size=input_size, grayscale=gray)
            postprocess(valset, size=input_size, grayscale=gray)

            # Get their numpy vectors to feed to lasagne
            x_train = images_to_vectors(trainset, input_size, num_dimensions=1 if gray else 3)
            x_val = images_to_vectors(valset, input_size, num_dimensions=1 if gray else 3)

            # Prepare theano tensors for classes and inputs
            input_var = T.tensor4('inputs')
            target_var = T.ivector('targets')

            neural_network = network(input_size, input_var)
            train_fn, val_fn, predict_fn = build_network(neural_network, input_var, target_var)
            training_loss, val_loss, val_acc = train(train_fn, val_fn, predict_fn, x_train, y_train, x_val, y_val,
                                                     num_epochs)

            if len(weights) <= 1:
                val_losses.append(val_loss)
                train_losses.append(training_loss)
                val_accs.append(val_acc)
            else:
                weighted_logloss = predict(predict_fn, x_val)
                print('Weighted logloss of fold: %f' % weighted_logloss)
                predictions.append(weighted_logloss)
        # Calculate weighted logloss for this fold
        if len(weights) > 1:
            val_losses.append(logloss(predictions, weights, y_val, num_classes=81))

    # Process metrics
    if len(weights) <= 1:
        mean_val_loss = np.mean(val_losses)
        mean_train_loss = np.mean(train_losses)
        mean_accuracy = np.mean(val_accs)
        std_val_loss = np.std(val_losses)
        std_val_acc = np.std(val_accs)
        std_train_loss = np.std(train_losses)

        print('Mean validation loss: %f (std: %f)' % (mean_val_loss, std_val_loss))
        print('Mean training loss: %f (std: %f)' % (mean_train_loss, std_train_loss))
        print('Mean training accuracy: %f (std %f)' % (mean_accuracy, std_val_acc))
    else:
        print('Weighted logloss: %f (std %f)' %(np.mean(val_losses), np.std(val_losses)))


# Train a single model. Evaluate using a validation set first to estimate where the model is going.
# Flip is the argument on which the model switches to the full dataset
def train_single_with_warmup(train_dir, test_dir, network=build_rgb_cnn, num_epochs=400, flip=200, input_size=45,
                             learning_rate=0.005, gray=False, augment=True):
    assert (num_epochs > flip)

    train_images = load_images(train_dir, is_train=True, permute=False)
    training_labels = list([img.label for img in train_images])
    classes_set = list(sorted(set(training_labels)))
    class_to_index = {key: index for index, key in enumerate(classes_set)}

    print('Building network...')
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    neural_network = network(input_size, input_var)
    print_network(neural_network)
    train_fn, val_fn, predict_fn = build_network(neural_network, input_var, target_var, learning_rate=learning_rate)

    if flip > 0:
        # Create validation and training set
        (trainset, valset) = split_special(train_images, 2, True)[0]  # Use the old validation way

        if augment:
            print("Augmenting trainset images")
            trainset = augmentation(trainset)
            print("Augmented to %d images" % len(trainset))

        print('Post processing train and validation set')
        postprocess(trainset, size=input_size, grayscale=gray)
        postprocess(valset, size=input_size, grayscale=gray)

        x_train = images_to_vectors(trainset, input_size, num_dimensions=1 if gray else 3)
        x_val = images_to_vectors(valset, input_size, num_dimensions=1 if gray else 3)

        for img in train_images:
            img.disposeImage()

        y_train = np.concatenate(np.array([[class_to_index[img.label] for img in trainset]], dtype=np.uint8))
        y_val = np.concatenate(np.array([[class_to_index[img.label] for img in valset]], dtype=np.uint8))
        print('Training %d iterations as a warmup' % flip)
        train(train_fn, val_fn, predict_fn, x_train, y_train, x_val, y_val, flip, show_validation=True)

    # Flip or fully train
    train_images = load_images(train_dir, is_train=True, permute=False)
    train_images = augmentation(train_images)
    postprocess(train_images, input_size)
    x_train = images_to_vectors(train_images, input_size)
    y_train = np.concatenate(np.array([[class_to_index[img.label] for img in train_images]], dtype=np.uint8))

    print('Training %d iterations on full set' % (num_epochs - flip))
    train(train_fn, val_fn, predict_fn, x_train, y_train,
          None if flip <= 0 else x_val, None if flip <= 0 else y_val,
          num_epochs if flip <= 0 else num_epochs - flip,
          show_validation=False)

    # Predict
    print('Predicting testset')
    test_images = sorted(load_images(test_dir, is_train=False, permute=False), key=lambda x: x.identifier)
    postprocess(test_images, size=input_size, grayscale=gray)
    x_test = images_to_vectors(test_images, input_size, num_dimensions=1 if gray else 3)
    predictions = predict(predict_fn, x_test)
    print('Writing to CSV...')
    write_csv(test_images, predictions, classes_set, filename='result_warmup.csv')
    print('Finished.')


def train_and_predict_ensemble(train_dir, test_dir, networks, weights, epochs, input_sizes, learning_rates, grays,
                               augment=True):
    assert (sum(weights) - 1 <= 0.001)

    train_images = load_images(train_dir, is_train=True, permute=False)
    training_labels = list([img.label for img in train_images])
    classes_set = list(sorted(set(training_labels)))
    class_to_index = {key: index for index, key in enumerate(classes_set)}

    test_images = None
    preds = []

    print('Loading training dataset')
    if augment:
        print('Augmenting images...')
        train_images = augmentation(train_images)
        print("Augmented to %d images" % len(train_images))

    i = 0
    for input_size, network, num_epochs, gray, learning_rate in zip(input_sizes, networks, epochs,
                                                                    grays, learning_rates):
        i += 1
        print('Start training next network')
        postprocess(train_images, size=input_size, grayscale=gray)

        print('Extracting feature vectors (dimensions: %d)' % (1 if gray else 3))
        x_train = images_to_vectors(train_images, input_size, num_dimensions=1 if gray else 3)

        # Wipe features to preserve memory
        for img in train_images:
            img.clearFeatures()
        gc.collect()

        y_train = np.concatenate(np.array([[class_to_index[img.label] for img in train_images]], dtype=np.uint8))

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        convnet = network(input_size, input_var)
        train_fn, val_fn, predict_fn = build_network(convnet, input_var, target_var, learning_rate=learning_rate)

        print('Training network...')
        train(train_fn, val_fn, predict_fn, x_train, y_train, None, None, num_epochs, show_validation=False)

        # Save network
        print('Saving model %d' % i)
        np.savez('model_%d.npz' % i, *lasagne.layers.get_all_param_values(convnet))

        # Now we predict the training set
        if test_images is None:
            print("Loading test images")
            test_images = sorted(load_images(test_dir, is_train=False, permute=False), key=lambda x: x.identifier)

        postprocess(test_images, size=input_size, grayscale=gray)
        x_test = images_to_vectors(test_images, input_size, num_dimensions=1 if gray else 3)
        preds.append(predict(predict_fn, x_test))
        print('Finished predictions for current network.')

        # Clear images from memory
        for img in test_images:
            img.disposeImage()
            img.clearFeatures()

        gc.collect()  # Force garbage collect

    if len(preds) > 1:
        # Now take a weighted sample of all nets
        predictions = np.zeros((len(test_images), len(class_to_index)))
        for w, pred in zip(weights, preds):
            predictions = predictions + (w * pred)
    else:
        predictions = preds[0]

    # Predictions
    write_csv(test_images, predictions, classes_set, combined=len(preds) > 1)
    print("Finished")


train_and_predict_ensemble(['data/train'],  ['data/test'],
                           networks=(build_rgb_cnn, build_grayscale_cnn),
                           learning_rates=(0.005, 0.005),
                           grays=(False, True),
                           input_sizes=(45, 45),
                           weights=(0.6, 0.4),
                           epochs=(300, 150),
                           augment=True)

# train_and_predict_ensemble(['data/train'],  ['data/test'],
#                            networks=(build_rgb_cnn, build_rgb_cnn_2, build_grayscale_cnn),
#                            learning_rates=(0.005, 0.005, 0.005),
#                            grays=(False, False, True),
#                            input_sizes=(45, 48, 45),
#                            weights=(0.5, 0.25, 0.25),
#                            epochs=(250, 200, 200),
#                            augment=True)

# train_single_with_warmup(['data/train'],  ['data/test'],
#                         build_rgb_cnn, 400, flip=200, input_size=45, learning_rate=0.005, gray=False, augment=True)

# Example of evaluating one model, explicitly takes tuples:
# Trains default RGB network:
# cross_validate(['data/train'],
#                networks=(build_rgb_cnn, ),
#                epochs=(10, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                grays=(False, ),
#                num_folds=2, augment=True)

# Trains grayscale network:
# cross_validate(['data/train'],
#                networks=(build_grayscale_cnn, ),
#                grays=(True, ),
#                epochs=(150, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                num_folds=2, augment=True)

# Trains 2nd RGB network
# cross_validate(['data/train'],
#                networks=(build_rgb_cnn_2, ),
#                grays=(False, ),
#                epochs=(150, ),
#                input_sizes=(45, ),
#                weights=(1, ),
#                learning_rates=(0.005, ),
#                num_folds=2, augment=True)

# Example of evaluating an ensemble 50/50
# cross_validate(['data/train'],
#                networks=(build_rgb_cnn, build_rgb_cnn_2),
#                epochs=(10, 10),
#                input_sizes=(45, 48),
#                weights=(0.5, 0.5),
#                learning_rates=(0.005, 0.005),
#                grays=(False, False),
#                num_folds=2, augment=True)