import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


def plot_filters_single_channel_big(t, place_to_store):
    # setting the rows and columns
    nrows = t.shape[0] * t.shape[2]
    ncols = t.shape[1] * t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols / 10, nrows / 200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)


def plot_filters_single_channel(t, place_to_store):
    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            plt.figure(figsize=(7, 7))
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            plt.imshow(npimg)
            name = str(i) + '-' + str(j)
            plt.axis('off')
            plt.savefig(f'{place_to_store}/{name}.png')
            plt.close()


def plot_filters_multi_channel(t, place_to_store):
    for i in range(t.shape[0]):
        plt.figure(figsize=(7, 7))
        npimg = np.array(t[i].numpy(), np.float32)
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        plt.imshow(npimg)
        plt.axis('off')
        name = str(i)
        plt.savefig(f'{place_to_store}/{name}.png')
        plt.close()


# place_to_store example: './multi' - you have to make such a folder before running.
# layer_num: index of convolutional layer you are interested in.
# single_channel = True works only in case on 3 channels (so gives out the RGB pictures).
def plot_weights(model, layer_num, place_to_store, single_channel=True, collated=False):
    # extracting the model features at the particular layer number
    layer = model.features[layer_num]

    # checking whether the layer is convolution layer or not
    if isinstance(layer, nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.features[layer_num].weight.data

        if single_channel:
            if collated:
                plot_filters_single_channel_big(weight_tensor, place_to_store)
            else:
                plot_filters_single_channel(weight_tensor, place_to_store)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor, place_to_store)
            else:
                print("Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")


def extract_filters(generator, layer_num):
    layer = generator.features[layer_num]
    if isinstance(layer, nn.Conv2d):
        t = generator.features[layer_num].weight.data
    filters = []
    for i in range(t.shape[0]):
        filter = t[i]
        filters.append(filter)

    return (filters)
