# Own
from image_loader import load, augment_images
from cross_validation import split_special
from preps import RotateTransform, SqueezeTransform, MirrorTransform, GaussianTransform, CropTransform

# Skimage
from skimage.transform import resize
from skimage import exposure, img_as_float
# OS
from os.path import basename
import os
import time

# Scientific
import numpy as np
import theano
import theano.tensor as T

# Neural
import lasagne


def images_to_vectors(imgs, size, num_dimensions=3):
    vect = np.zeros((len(imgs), num_dimensions, size, size), dtype=np.float32)  # Assume 3 channels
    for idx, img in enumerate(imgs):
        if num_dimensions > 1:
            rolled = np.rollaxis(img.image.astype(np.float32), 2, 0)
            vect[idx, :, :, :] = rolled
        else:
            vect[idx, 0, :, :] = img.image.astype(np.float32)
    return vect


def load_images(directories, is_train=False, permute=True):
    return load(directories, is_train, permute)


def postprocess(imgs, size, normalize=False):
    # Mass-resize them and convert to grayscale
    print("Postprocessing images and resize (at %d)" % size)
    for img in imgs:
        # preprocess using histogram equalization
        floatimg = img_as_float(img.image)
        dimensions = len(floatimg.shape)
        if normalize:
            if dimensions == 3:
                floatimg[:, :, 0] = exposure.equalize_hist(floatimg[:, :, 0])
                floatimg[:, :, 1] = exposure.equalize_hist(floatimg[:, :, 1])
                floatimg[:, :, 2] = exposure.equalize_hist(floatimg[:, :, 2])
            else:  # Grayscale
                floatimg[:, :] = exposure.equalize_hist(floatimg)
        img.image = resize(floatimg, (size, size))  # expect to return floats


def augmentation(images):
    transforms = list([RotateTransform(degrees) for degrees in [-10, -7.0, 7.0, 10]]) + \
                     [SqueezeTransform(), MirrorTransform()] # , GaussianTransform(sigma=3, multichannel=True)
    return augment_images(images, transforms)


def load_train_dataset(train_dir, test_dir, train_image_size=48, augment=True):
    # Loading
    train_images = load_images(train_dir, is_train=True, permute=False)

    # Create validation and training set
    (trainset, valset) = split_special(train_images, 2, True)[0]  # Use the old validation way

    if augment:
        print("Augmenting trainset images")
        trainset = augmentation(trainset)
        print("Augmented to %d images" % len(trainset))

    # Postprocess images
    postprocess(trainset, size=train_image_size)
    postprocess(valset, size=train_image_size)

    X_train = images_to_vectors(trainset, train_image_size)
    X_val = images_to_vectors(valset, train_image_size)

    for img in train_images:
        img.disposeImage()

    training_labels = list([img.label for img in train_images])
    classes_set = list(sorted(set(training_labels))) # USe sorted such that it is always thesame
    class_to_index = {key: index for index, key in enumerate(classes_set)}
    y_train = np.concatenate(np.array([[class_to_index[img.label] for img in trainset]], dtype=np.uint8))
    y_val = np.concatenate(np.array([[class_to_index[img.label] for img in valset]], dtype=np.uint8))

    return X_train, y_train, X_val, y_val, classes_set


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
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
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
    # As a bonus, also create an expression for the classification accuracy:
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


def train(train_fn, val_fn, X_train, y_train, X_val, y_val, num_epochs=500, input_size=45):
        print("Starting training...")

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(X_train, y_train, 256, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 256, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            val_loss = val_err / val_batches
            validation_acc = val_acc / val_batches * 100
            training_loss = train_err / train_batches

            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(training_loss))
            print("  validation loss:\t\t{:.6f}".format(val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(validation_acc))
        print('Finished %d iterations' % num_epochs)

def predict(predict_fn, x_test):
    return predict_fn(x_test)

def write_csv(test_images, predictions, id_to_class, filename='result.csv'):
    file = open(filename, 'w')
    file.write('Id,%s\n' % str.join(',', id_to_class))
    for idx, img in enumerate(test_images):
        probs = predictions[idx]
        thesum = np.sum(probs)
        if abs(thesum - 1) > 0.01:
            print('Warning: Incorrect probabilities for %d' % img.identifier)
        file.write('%d,%s\n' % (img.identifier, str.join(',', [('%.13f' % p) for p in probs])))
        # print("%d, (len:%d): %s" % (identifier, len(probs), str.join(',', [str(p) for p in probs])))

    file.close()


def train_and_predict(train_dir, test_dir, num_epochs=500, input_size=45, flipover=250):

    print('Loading dataset')
    x_train, y_train, x_val, y_val, id_to_class = load_train_dataset(train_dir, test_dir, input_size)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_rgb_cnn(input_size, input_var)
    train_fn, val_fn, predict_fn = build_network(network, input_var, target_var)

    if flipover > 0:
        train(train_fn, val_fn, x_train, y_train, x_val, y_val, flipover, input_size)
        print("Flipping over dataset to full trainset")
        train_images = load_images(train_dir, is_train=True, permute=False)
        train_images = augmentation(train_images)
        postprocess(train_images, input_size)
        x_train = images_to_vectors(train_images, input_size)
        y_train = np.concatenate(np.array([[id_to_class.index(img.label) for img in train_images]], dtype=np.uint8))
        train(train_fn, val_fn, x_train, y_train, x_val, y_val, num_epochs - flipover, input_size)
    else:
        train(train_fn, val_fn, x_train, y_train, x_val, y_val, num_epochs, input_size)

    # Prediction
    print("Starting evaluation of testset")
    test_images = sorted(load_images(test_dir, is_train=False, permute=False), key=lambda x: x.identifier)
    postprocess(test_images, size=input_size)

    x_test = images_to_vectors(test_images, input_size)
    predictions = predict(predict_fn, x_test)
    write_csv(test_images, predictions, id_to_class)
    print("Finished")

train_and_predict(['data/train'], ['data/test'], num_epochs=40, input_size=45, flipover=20)