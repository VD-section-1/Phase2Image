import os
from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, Input, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot
from numpy import asarray, zeros, ones
from numpy.random import randint
from random import random
from tqdm import tqdm


def define_discriminator(image_shape):
    """
    Define the discriminator model

    :param image_shape: shape of the input images
    :return: the discriminator model
    """

    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss="mse", optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

    return model


def resnet_block(n_filters, input_layer):
    """
    Define a resnet block

    :param n_filters: number of filters
    :param input_layer: input layer
    :return: the resnet block
    """

    init = RandomNormal(stddev=0.02)

    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    g = Concatenate()([g, input_layer])

    return g


def define_generator(image_shape, n_resnet=9):
    """
    Define the generator model

    :param image_shape: shape of the input images
    :param n_resnet: number of resnet blocks
    :return: the generator model
    """

    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    g = Conv2D(64, (7, 7), padding="same", kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    for _ in range(n_resnet):
        g = resnet_block(256, g)

    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation("tanh")(g)
    model = Model(in_image, out_image)

    return model


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    """
    Define the composite model

    :param g_model_1: the first generator model
    :param d_model: the discriminator model
    :param g_model_2: the second generator model
    :param image_shape: shape of the input images
    :return: the composite model
    """

    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    output_f = g_model_2(gen1_out)

    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=["mse", "mae", "mae", "mae"], loss_weights=[1, 5, 10, 10], optimizer=opt)

    return model


def generate_real_samples(dataset, n_samples, patch_shape):
    """
    Generate real samples

    :param dataset: dataset from which to generate the samples
    :param n_samples: number of samples to generate
    :param patch_shape: shape of the patches
    :return: the generated samples
    """

    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))

    return X, y


def generate_fake_samples(g_model, dataset, patch_shape):
    """
    Generate fake samples

    :param g_model: the generator model
    :param dataset: dataset from which to generate the samples
    :param patch_shape: shape of the patches
    :return: the generated samples
    """

    X = g_model.predict(dataset)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y


def save_models(step, g_model_AtoB, g_model_BtoA):
    """
    Save the models

    :param step: current step
    :param g_model_AtoB: the AtoB generator model
    :param g_model_BtoA: the BtoA generator model
    """

    path = "../data/model"
    if not os.path.exists(path):
        os.makedirs(path)

    filename1 = "g_model_AtoB_%06d.h5" % (step + 1)
    path1 = os.path.join(path, filename1)
    g_model_AtoB.save(path1)

    filename2 = "g_model_BtoA_%06d.h5" % (step + 1)
    path2 = os.path.join(path, filename2)
    g_model_BtoA.save(path2)

    print(">Saved: %s and %s" % (path1, path2))


def summarize_performance(step, g_model, trainX, name, n_samples=5):
    """
    Summarize the performance of the model

    :param step: current step
    :param g_model: the generator model
    :param trainX: training dataset
    :param name: name of the model
    :param n_samples: number of samples to summarize
    """

    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)

    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0

    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X_in[i])

    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis("off")
        pyplot.imshow(X_out[i])

    filename1 = "%s_generated_plot_%06d.png" % (name, (step + 1))

    path = "../data/output"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, filename1)
    pyplot.savefig(path)
    pyplot.close()


def update_image_pool(pool, images, max_size=50):
    """
    Update the pool of images

    :param pool: the pool of images
    :param images: the images to update the pool with
    :param max_size: maximum size of the pool
    :return: the updated pool
    """

    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image

    return asarray(selected)


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=50, batch=1):
    """
    Train the models

    :param d_model_A: the A discriminator model
    :param d_model_B: the B discriminator model
    :param g_model_AtoB: the AtoB generator model
    :param g_model_BtoA: the BtoA generator model
    :param c_model_AtoB: the AtoB composite model
    :param c_model_BtoA: the BtoA composite model
    :param dataset: training dataset
    :param epochs: number of epochs
    :param batch: batch size
    """

    n_patch = d_model_A.output_shape[1]
    trainA, trainB = dataset
    poolA, poolB = list(), list()
    bat_per_epo = int(len(trainA) / batch)
    n_steps = bat_per_epo * epochs

    for i in tqdm(range(n_steps)):
        X_realA, y_realA = generate_real_samples(trainA, batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, batch, n_patch)

        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

    print(
        ">%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]"
        % (i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
    )

    summarize_performance(i, g_model_AtoB, trainA, "AtoB")
    summarize_performance(i, g_model_BtoA, trainB, "BtoA")
    save_models(i, g_model_AtoB, g_model_BtoA)
