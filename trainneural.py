# Own
from image_loader import load, augment_images
from cross_validation import split_special
from preps import RotateTransform, SqueezeTransform, MirrorTransform

# Skimage
from skimage.transform import resize
from skimage import color

# OS
from os.path import basename
import os
import time

#Scientific
import numpy as np
import theano
import theano.tensor as T

# Neural
import lasagne

def images_to_vectors(imgs, size):
        vect = np.zeros((len(imgs), 1, size, size), dtype=np.float32)  # Assume 1 channel
        for idx, img in enumerate(imgs):
            vect[idx, 0, :, :] = img.image.astype(np.float32)
        return vect


def load_and_augment(directories, is_train=False, permute=True, augment=False):
        imgs = load(directories, is_train, permute)
        if augment:
            print("Augmenting images")
            transforms = list([RotateTransform(degrees) for degrees in [-10, -7.0, 7.0, 10]]) + \
               [SqueezeTransform(), MirrorTransform()]
            imgs = augment_images(imgs, transforms)
            print("Augmented to %d images" % len(imgs))
        return imgs


def postprocess(imgs, size):
     # Mass-resize them and convert to grayscale
    print("Postprocessing images to grayscale and resize (at %d)" % size)
    for img in imgs:
            img.image = resize(color.rgb2gray(img.image), (size, size))  # expect to return floats


def load_train_dataset(train_dir, test_dir, train_image_size=42):

    # Loading
    train_images = load_and_augment(train_dir, is_train=True, permute=False, augment=False)

    # Create validation and training set
    (trainset, valset) = split_special(train_images, 2, True)[0]  # Use the old validation way

    # Postprocess images
    postprocess(train_images, size=train_image_size)

    X_train = images_to_vectors(trainset, train_image_size)
    X_val = images_to_vectors(valset, train_image_size)

    # Wipe the image copies from memory
    for img in train_images:
        img.disposeImage()

    training_labels = list([img.label for img in train_images])
    classes_set = list(set(training_labels))
    class_to_index = {key: index for index, key in enumerate(classes_set)}
    y_train = np.concatenate(np.array([[class_to_index[img.label] for img in trainset]], dtype=np.uint8))
    y_val = np.concatenate(np.array([[class_to_index[img.label] for img in valset]], dtype=np.uint8))

    return X_train, y_train, X_val, y_val, classes_set


def build_cnn(input_size, input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, input_size, input_size), input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
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


def train_and_predict(train_dir, test_dir, num_epochs=500, input_size=42):
    # Load the dataset
    print("Loading training data...")
    X_train, y_train, X_val, y_val, id_to_class = load_train_dataset(train_dir, test_dir, input_size)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_size, input_var)

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
            loss, params, learning_rate=0.01, momentum=0.9)

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

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
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
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


    # Prediction
    print("Starting evaluation of testset")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)

    print("Loading test images")

    test_images = load_and_augment(test_dir, is_train=False, permute=False, augment=False)
    postprocess(test_images, size=input_size)

    for img in test_images[1:3]:
        identifier = int(os.path.splitext(basename(img.filename))[0])
        test_data = np.reshape(img.image, (1, input_size, input_size))
        predictions = predict_fn(test_data)
        print("%d, (len:%d): %s" % (identifier, len(predictions), str.join(',', predictions)))

    print("Finished")

train_and_predict(['data/train'], ['data/test'], num_epochs=10, input_size=42)
    # file = open('result.csv', 'w')
    # file.write('Id,%s\n' % str.join(',', id_to_class))
    # for img in test_images:
    #     test_data = np.reshape(img.image, (1, input_size, input_size))
    #     predictions = predict_fn(test_data)
    #
    #     identifier = int(os.path.splitext(basename(image.filename))[0])
    #     file.write('%d,%s\n' % (identifier, str.join(',', [('%.13f' % p) for p in predictions[0]])))
    # file.close()
    #
    # # After training, we compute and print the test error:
    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

