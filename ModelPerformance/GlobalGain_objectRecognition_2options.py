import numpy as np
from keras.models import model_from_json, Model
from keras.layers import Dense
from keras import optimizers
from asn.arousal.layers import ASNTransfer_arousal
from asn.arousal.utils import insert_layer_nonseq
from asn.utils import load_pickle
from asn.layers.training import ASNTransfer
from asn.evaluation.generators import coco_squared
from sklearn.metrics import roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from asn.training.callbacks import LossHistory_binary
from keras.callbacks import ModelCheckpoint, CSVLogger
from asn.resnets.resnet_asn import ResnetBuilder

def make_subDataset(cats, dataset_in):
    trials = []

    # find minimum number of images per cat
    n = int(np.min(np.sum(dataset_in['y'],axis=0)[cats]))

    y = np.zeros((len(cats) * n, len(cats)))

    for i, c in enumerate(cats):
        # collect the images per cat
        tmp = np.where(dataset_in['y'][:, c])[0]
        np.random.shuffle(tmp)
        trials.extend(tmp[:n])

        y[i * n: (i+1) * n, i] = 1

    idx = np.arange(len(trials))
    # shuffle
    np.random.shuffle(idx)

    # assign to a new dataset
    dataset_out = {}
    dataset_out['y'] = y[idx]
    if type(dataset_in['x_ids'][0]) == str:
        dataset_out['x_ids'] = [dataset_in['x_ids'][t] for t in np.array(trials)[idx]]
    else:
        dataset_out['x_ids'] = np.array([dataset_in['x_ids'][t] for t in np.array(trials)[idx]])
    dataset_out['dataType'] = dataset_in['dataType']
    return dataset_out


def insert_layer_factory():
    return ASNTransfer_arousal(loc='out')

new_tracker = False
train = True
test = True
filename = 'Results/tracker_ObjectRecognition_2options.pkl'
if new_tracker == True:
    tracker = {}
else:
    tracker = joblib.load(filename)

np.random.seed(3)
for dataset in ['1_street','2_food']:
    if dataset not in tracker:
        tracker[dataset] = {}

    #  Load the dataset
    img_dir_train = '/mnt/Googleplex2/coco/images/' + dataset + '/train2017_multi_radial/'
    img_dir_test = '/mnt/Googleplex2/coco/images/' + dataset + '/val2017_single_radial/'

    dataset_base = load_pickle(
                    '/mnt/Googleplex2/coco/datasets/' + dataset + '/train2017_multi_radial.pickle')
    dataset_test = load_pickle(
                    '/mnt/Googleplex2/coco/datasets/' + dataset + '/val2017_single_radial.pickle')

    batch_size = 50

    proportion = 0.2
    idx = np.arange(len(dataset_base['x_ids']))
    np.random.shuffle(idx)
    dataset_val = {}
    dataset_val['x_ids'] = dataset_base['x_ids'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion))]
    dataset_val['y'] = dataset_base['y'][idx][:int(np.ceil(len(dataset_base['x_ids']) * proportion))]

    dataset_train = {}
    dataset_train['x_ids'] = dataset_base['x_ids'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion)):]
    dataset_train['y'] = dataset_base['y'][idx][int(np.ceil(len(dataset_base['x_ids']) * proportion)):]
    dataset_val['dataType'] = 'train2017'
    dataset_train['dataType'] = 'train2017'

    #  Load the pre-trained model
    learning_mode = 'constant'
    optimizer = 'Adam'
    optimizer_func = eval('optimizers.' + optimizer)
    lr = 1e-4

    # Make datasets with only 4 randomly chosen categories and repeat for some iterations
    iterations = 20
    gains = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06,
                 1.08, 1.1, 1.12,
                 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4, 1.42, 1.44, 1.46, 1.48, 1.5]
    if new_tracker == True:
        tracker['gains'] = gains

    tracker['iterations'] = iterations

    for i in range(iterations):

        if 'iter_' + str(i) not in tracker[dataset]:
            #  Load the pre-trained model
            modelBase = ResnetBuilder.build_resnet_18_v0((224, 224, 3), 1000)
            modelBase.summary()

            weight_dir = '/mnt/Googleplex2/PycharmProjects/SpatialAttention_asn/ModelTraining/Training_imagenet/LISA_GPUs/Results/'

            weight_path_resnet = weight_dir + 'weights_training1/FinalWeights.h5'
            modelBase.load_weights(weight_path_resnet)

            temp = Dense(2, activation='sigmoid')(modelBase.layers[-2].output)

            modelTrain = Model(inputs=modelBase.input, outputs=temp)

            for l in range(len(modelTrain.layers)):
                if modelTrain.layers[l].name.startswith('dense'):
                    modelTrain.layers[l].trainable = True
                else:
                    modelTrain.layers[l].trainable = False

            del modelBase

            tracker[dataset]['iter_' + str(i)] = {}
            np.random.seed(i)

            # pick 2 of the 8 categories.
            cats = np.arange(8)
            np.random.shuffle(cats)
            cats_i = cats[:2]
            tracker[dataset]['iter_' + str(i)]['categories'] = cats_i

            dataset_test_sub = make_subDataset(cats_i, dataset_test)
            gen_test = coco_squared(dataset_test_sub, batch_size=len(dataset_test_sub['x_ids']) - 1, img_dir=img_dir_test)
            (x_test, y_test) = next(gen_test)

            dataset_val_sub = make_subDataset(cats_i, dataset_val)
            gen_val = coco_squared(dataset_val_sub, batch_size=len(dataset_val_sub['x_ids'])-1, img_dir=img_dir_train)
            (x_val, y_val) = next(gen_val)

            dataset_train_sub = make_subDataset(cats_i, dataset_train)
            gen_train = coco_squared(dataset_train_sub, batch_size=len(dataset_train_sub['x_ids']), img_dir=img_dir_train)
            (x_train, y_train) = next(gen_train)

            modelTrain.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                metrics=['binary_accuracy'])

            weight_path_new = 'Results/'+ dataset +'_2options_Iteration' + str(i)

            if train == True:
                batch_size = 50
                epochs = 75

                checkpointer = ModelCheckpoint(filepath=weight_path_new + '.h5', monitor='val_loss', verbose=1,
                                               save_best_only=True)
                history_call = LossHistory_binary()

                logger = CSVLogger('Results/' + dataset + '_2options_Iteration' + str(i) +'.csv')

                callbacks = [checkpointer, history_call, logger]

                modelTrain.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                        metrics=['binary_accuracy'])

                history = modelTrain.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                              validation_data=(x_val, y_val), shuffle=True,
                                              callbacks=callbacks)
                tracker[dataset]['iter_' + str(i)]['Training'] = history.history

            # Reload the best weights
            modelTrain.load_weights(weight_path_new + '.h5')

            modelArousal = insert_layer_nonseq(modelTrain, 'asn_transfer_', insert_layer_factory,
                                           insert_layer_name='Arousal', position='replace',
                                           additionalInput=True)

            modelArousal.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr), metrics=['binary_accuracy'])

            loss = np.zeros(len(gains))
            acc = np.zeros(len(gains))
            auc = np.zeros(len(gains))
            tracker[dataset]['iter_' + str(i)]['Test'] = {}
            for g, gain in enumerate(gains):
                print('Gain: ' + str(gain))
                modelArousal.layers[2].set_weights(np.array([[gain]]))
                loss[g], acc[g] = modelArousal.evaluate(x_test, y_test, verbose=False)
                auc[g] = roc_auc_score(y_test, modelArousal.predict(x_test, verbose=False))
                print('AUC: ' + str(auc))

            tracker[dataset]['iter_' + str(i)]['Test']['binaryCE'] = loss
            tracker[dataset]['iter_' + str(i)]['Test']['binaryAcc'] = acc
            tracker[dataset]['iter_' + str(i)]['Test']['AUC'] = auc

            joblib.dump(tracker, filename, compress=True)

