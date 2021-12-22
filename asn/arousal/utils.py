import re
import numpy as np
from asn.arousal.layers import ArousalValue
from keras.models import Model

def makeImageNoImageDataset(x, reshuffle=True, proportion=0.5, mode='random', num_rand=25):
    print('Making dataset with ' + str(num_rand) + ' images averaged.')
    idx = np.arange(len(x))
    if reshuffle == True:
        np.random.shuffle(idx)

    prop_idx = int(np.floor(len(x) * proportion))

    x_keep = x[idx][:prop_idx]

    x_recycle = x[idx][prop_idx:]
    del x

    x_new = np.zeros(x_recycle.shape)

    if mode == 'random':
        # pick a random amount of images & combine
        for i in range(len(x_recycle)):
            if num_rand < 2:
                num_rand_array = 2
            else:
                num_rand_array = num_rand
            # draw 24 other images from x_recycle
            rand_imgs = np.random.randint(0, len(x_recycle), num_rand_array)
            if num_rand < 2:
                # This allows for mixtures between 1 & 2
                x_new[i] = np.sum([x_recycle[rand_imgs[0]], (num_rand - 1) * x_recycle[rand_imgs[1]]], axis=0)[
                           np.newaxis, :, :, :] / num_rand
            else:
                x_new[i] = np.mean(x_recycle[rand_imgs], axis=0)[np.newaxis, :, :, :]


    elif mode == 'gray':
        x_new = np.ones(x_recycle.shape) * 0.5
    else:
        raise ValueError('Mode is not implemented. Try random or gray')

    x = np.concatenate([x_keep, x_new], axis=0)

    y = np.zeros((len(x), 1))
    y[:prop_idx] = 1

    # reshuffle both at the end
    np.random.shuffle(idx)

    return x[idx], y[idx]

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after', additionalInput=None):
    """
    adapted from https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
    :param model:
    :param layer_regex:
    :param insert_layer_factory:
    :param insert_layer_name:
    :param position:
    :return:
    """
    # collect all current nodes
    # from: https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/layer_utils.py
    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for l, layer in enumerate(model.layers):
        for node in layer._outbound_nodes:
            if node in relevant_nodes:
                layer_name = node.outbound_layer.name

                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update({layer_name: [layer.name]})
                else:
                    if layer.name not in network_dict['input_layers_of'][layer_name]:
                        network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
                {model.layers[0].name: model.input})

    count = 0
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.findall(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            if (layer.name.startswith('batch_norm')) & (position == 'replace'):
                # Replace the batch norm layer with a static version of it but inhereting the old parameters
                new_layer = insert_layer_factory(layer)
            else:
                new_layer = insert_layer_factory()

            if insert_layer_name:
                new_layer.name = insert_layer_name + '_' + str(count)
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)

            if new_layer.name.startswith('AlphaDrop') | new_layer.name.startswith('Drop'):
                training = True
            else:
                training = False

            if (additionalInput is not None):
                if count == 0:
                    y = ArousalValue((1,))([])
                    x = new_layer([x,y])
                else:
                    x = new_layer([x,y])
            else:
                x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)#, training = training)

            # for naming
            count = count + 1

        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)