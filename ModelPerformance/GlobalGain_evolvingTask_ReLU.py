import sys
sys.path.extend(['/mnt/Googleplex2/PycharmProjects/Arousal_asn','/home/lynn/miniconda3/envs/tensorflow_1.12_arousal2/lib/python3.6/site-packages/asn'])
#               ])

import numpy as np
from keras.models import model_from_json, Model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from asn.arousal.layers import ReLU_arousal
from asn.arousal.utils import insert_layer_nonseq
from asn.utils import load_pickle
#from asn.layers.training import ASNTransfer
from asn.evaluation.generators import coco_squared
from asn.evaluation.metrics import compute_SDTmeasures
from asn.training.callbacks import LossHistory_binary
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from classification_models.keras import Classifiers

def makeImageNoImageDataset(x, reshuffle = True, proportion = 0.5, mode='random', num_rand=25):

    print('Making dataset with ' + str(num_rand) + ' images averaged.')
    idx = np.arange(len(x))
    if reshuffle == True:
        np.random.shuffle(idx)

    prop_idx = int(np.floor(len(x) * proportion))

    x_keep = x[idx][:prop_idx]
    #y_keep = y[idx][:(len(x) * proportion)]

    x_recycle = x[idx][prop_idx:]
    #y_recycle = y[idx][(len(x) * proportion):]
    del x

    x_new = np.zeros(x_recycle.shape)

    if mode == 'random':
        # pick a random amount of images & combine
        for i in range(len(x_recycle)):
            num_rand_array = int(np.ceil(num_rand))
            # draw other images from x_recycle
            rand_imgs = np.random.randint(0, len(x_recycle),  num_rand_array)

            if np.divmod(num_rand, 1) != 0:
                # This allows for mixtures
                full_images, part_images = np.divmod(num_rand, 1)
                x_new[i] = (np.sum(x_recycle[rand_imgs[:int(full_images)]], axis=0) + part_images * x_recycle[rand_imgs[int(full_images)]])[np.newaxis, :,:,:] /num_rand

            else:
                x_new[i] = np.mean(x_recycle[rand_imgs], axis=0)[np.newaxis, :,:,:]


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


def insert_layer_factory():
    return ReLU_arousal(loc='out')

ResNet18, preprocess_input = Classifiers.get('resnet18')

multi_gpu = False
train = False
test = True
np.random.seed(3)

dataset = '1_street'

# Create a simple task: distinguish images from non-images
img_dir_train = '/mnt/Googleplex2/coco/images/' + dataset + '/train2017_multi_radial/'
img_dir_test = '/mnt/Googleplex2/coco/images/' + dataset + '/val2017_single_radial/'

dataset_base = load_pickle(
                    '/mnt/Googleplex2/coco/datasets/' + dataset + '/train2017_multi_radial.pickle')

# Divide in train, val and test set.
proportion_test = 0.2
proportion_val = 0.3

idx = np.arange(len(dataset_base['x_ids']))
np.random.shuffle(idx)
dataset_test = {}
dataset_test['x_ids'] = dataset_base['x_ids'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion_test))]
dataset_test['y'] = dataset_base['y'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion_test))]
dataset_test['dataType'] = 'train2017'
gen_test = coco_squared(dataset_test, batch_size=len(dataset_test['x_ids']) - 1, img_dir=img_dir_train)


dataset_base2 = {}
dataset_base2['x_ids'] = dataset_base['x_ids'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion_test)):]
dataset_base2['y'] = dataset_base['y'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion_test)):]

dataset_val = {}
dataset_val['x_ids'] = dataset_base2['x_ids'][:int(np.ceil(len(dataset_base2['x_ids']) * proportion_val))]
dataset_val['y'] = dataset_base2['y'][:int(np.ceil(len(dataset_base2['x_ids']) * proportion_val))]
dataset_val['dataType'] = 'train2017'
gen_val = coco_squared(dataset_val, batch_size=len(dataset_val['x_ids'])//4, img_dir=img_dir_train)


chunks = 10 # for loading
dataset_train= {}
dataset_train['x_ids'] = dataset_base2['x_ids'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['y'] = dataset_base2['y'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['dataType'] = 'train2017'
gen_train = coco_squared(dataset_train, batch_size=len(dataset_train['x_ids'])//chunks, img_dir=img_dir_train)

del dataset_base, dataset_base2

mode = 'random'
trainings = 10
optimizer = 'SGD'
optimizer_func = eval('optimizers.' + optimizer)
lr = 1e-4

num_rands = [1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8, 10]
new_tracker =False
filename = 'Results/tracker_BinaryEvolvingTask_ReLU.pkl'
if new_tracker == True:
    tracker = {}
    tracker['num_rands'] = num_rands
else:
    tracker = joblib.load(filename)
if train == True:
    for num_rand in num_rands:
        print(num_rand)
        if mode == 'gray':
            tag = mode
        else:
            tag = mode + str(num_rand)
        if tag not in tracker:
            tracker[tag] = {}

        for t in range(trainings):
            if 'Training-' + str(t) not in tracker[tag]:
                tracker[tag]['Training-' + str(t)]={}

            if 'History' not in tracker[tag]['Training-' + str(t)]:
                if 'x_val_binary' not in locals():
                    for p in range(4):
                        print('Validation: ' + str(p))
                        # print(x_binary.shape)
                        (x_val, y_val) = next(gen_val)

                        x_val_n, y_val_n = makeImageNoImageDataset(x_val, mode=mode, num_rand=num_rand)

                        x_val_n = preprocess_input(x_val_n * 255)

                        if p == 0:
                            x_val_binary = x_val_n
                            y_val_binary = y_val_n
                        else:
                            x_val_binary = np.concatenate([x_val_binary, x_val_n], axis=0)
                            y_val_binary = np.concatenate([y_val_binary, y_val_n], axis=0)

                        del x_val_n, y_val_n, x_val, y_val

                if 'x_binary' not in locals():
                    for p in range(chunks):
                        print('Training: ' + str(p))

                        (x_train, y_train) = next(gen_train)

                        x_train_n, y_train_n = makeImageNoImageDataset(x_train, mode=mode, num_rand=num_rand)

                        x_train_n = preprocess_input(x_train_n * 255)

                        if p == 0:
                            x_binary = x_train_n
                            y_binary = y_train_n
                        else:
                            x_binary = np.concatenate([x_binary, x_train_n], axis=0)
                            y_binary = np.concatenate([y_binary, y_train_n], axis=0)

                        del x_train, y_train, x_train_n, y_train_n

                np.random.seed(t)
                modelBase = ResNet18((224, 224, 3), weights='imagenet')

                temp = Dense(1, activation='sigmoid', name='dense')(modelBase.layers[-3].output)

                modelBinary = Model(inputs=modelBase.input, outputs=temp)


                for l in range(len(modelBinary.layers)):
                    if modelBinary.layers[l].name.startswith('dense'):
                        modelBinary.layers[l].trainable = True
                    else:
                        modelBinary.layers[l].trainable = False

                modelBinary.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr, momentum=0.9),
                                    metrics=['binary_accuracy'])
                del modelBase
                weight_path_new = 'Results/EvolvingTask_ReLU/BinaryEvolvingTask_' + tag + '_' + str(t)

                if train == True:
                    batch_size = 50
                    epochs = 50

                    checkpointer = ModelCheckpoint(filepath=weight_path_new + '.h5', monitor='val_loss', verbose=1,
                                                   save_best_only=True)
                    history_call = LossHistory_binary()

                    logger = CSVLogger(
                        'Results/BinaryEvolvingTask_log_' + tag + '_' + str(t) + '.csv')

                    callbacks = [checkpointer, history_call, logger]

                    if multi_gpu == True:

                        model_gpus = multi_gpu_model(modelBinary, gpus=2)

                        model_gpus.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr, momentum=0.9),
                                           metrics=['binary_accuracy'])

                        history = model_gpus.fit(x_binary, y_binary, batch_size=batch_size, epochs=epochs,
                                                 validation_data=(x_val_binary, y_val_binary), shuffle=True,
                                                 callbacks=callbacks)

                    else:
                        modelBinary.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr, momentum=0.9),
                                            metrics=['binary_accuracy'])

                        history = modelBinary.fit(x_binary, y_binary, batch_size=batch_size, epochs=epochs,
                                                  validation_data=(x_val_binary, y_val_binary), shuffle=True,
                                                  callbacks=callbacks)

                    tracker[tag]['Training-' + str(t)]['History'] = history.history
                    joblib.dump(tracker, filename, compress=True)

        if 'x_binary' in locals():
            del x_binary, x_val_binary, y_binary, y_val_binary

if test == True:
    # Test the weights with varying levels of gain.

    modelBase = ResNet18((224, 224, 3), weights='imagenet')
    modelBase.summary()

    temp = Dense(1, activation='sigmoid')(modelBase.layers[-3].output)

    modelBinary = Model(inputs=modelBase.input, outputs=temp)

    modelBinary.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr, momentum=0.9),
                        metrics=['binary_accuracy'])
    del modelBase

    gains = np.round(np.arange(0.8, 1.21, 0.01), 2)

    tracker['gains'] = gains

    for num_rand in num_rands:

        if mode == 'gray':
            tag = mode
        else:
            tag = mode + str(num_rand)
        if tag not in tracker:
            tracker[tag] = {}
        for t in range(trainings):
           if 'Training-' + str(t) not in tracker[tag]:
               tracker[tag]['Training-' + str(t)] = {}

           if 'Test' not in tracker[tag]['Training-' + str(t)]:
               print(num_rand)
               if 'x_test_binary' not in locals():

                   (x_test, y_test) = next(gen_test)
                   x_test_binary, y_test_binary = makeImageNoImageDataset(x_test, mode=mode, num_rand=num_rand)
                   x_test_binary = preprocess_input(x_test_binary * 255)

                   if type(x_test_binary) == list:
                       x_test_binary = np.array(x_test_binary)
                   del x_test, y_test

               print('Testing training ' + str(t))
               weight_path_new = 'Results/EvolvingTask_ReLU/BinaryEvolvingTask_' + tag + '_' + str(
                   t)
               # reload the best weights from the training:
               modelBinary.load_weights(weight_path_new + '.h5')

               modelArousal = insert_layer_nonseq(modelBinary, 'relu', insert_layer_factory,
                                                  insert_layer_name='Arousal', position='replace',
                                                  additionalInput=True)

               modelArousal.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                    metrics=['binary_accuracy'])

               tracker[tag]['Training-' + str(t)]['Test'] = {}

               loss = np.zeros(len(gains))
               acc = np.zeros(len(gains))
               d = np.zeros(len(gains))
               c = np.zeros(len(gains))
               beta = np.zeros(len(gains))
               hit_rate = np.zeros(len(gains))
               fa_rate = np.zeros(len(gains))

               for g, gain in enumerate(gains):
                   print('Gain: ' + str(gain))
                   # identify arousal value layer
                   layer_name = [layer.name for layer in modelArousal.layers if layer.name.startswith('arousal_value_')]

                   modelArousal.get_layer(layer_name[0]).set_weights(np.array([[gain]]))
                   loss[g], acc[g] = modelArousal.evaluate(x_test_binary, y_test_binary, verbose=False)
                   preds = modelArousal.predict(x_test_binary, verbose=False)
                   sd_params = compute_SDTmeasures(preds, y_test_binary, adjusted=True)
                   d[g] = sd_params['dprime']
                   c[g] = sd_params['c']
                   beta[g] = sd_params['beta']
                   hit_rate[g] = sd_params['hit_rate']
                   fa_rate[g] = sd_params['fa_rate']
                   print('Acc: ' + str(acc))

               tracker[tag]['Training-' + str(t)]['Test']['binaryCE'] = loss
               tracker[tag]['Training-' + str(t)]['Test']['binaryAcc'] = acc
               tracker[tag]['Training-' + str(t)]['Test']['dPrime'] = d
               tracker[tag]['Training-' + str(t)]['Test']['criterion'] = c
               tracker[tag]['Training-' + str(t)]['Test']['beta'] = beta
               tracker[tag]['Training-' + str(t)]['Test']['hit-rate'] = hit_rate
               tracker[tag]['Training-' + str(t)]['Test']['fa-rate'] = fa_rate

               joblib.dump(tracker, filename, compress=True)

        if 'x_test_binary' in locals():
            del x_test_binary, y_test_binary

