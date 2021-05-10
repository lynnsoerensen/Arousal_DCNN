import numpy as np
from keras.models import model_from_json
from keras import optimizers
from asn.arousal.layers import ASNTransfer_arousal
from asn.arousal.utils import insert_layer_nonseq
from asn.utils import load_pickle
from asn.layers.training import ASNTransfer
from asn.evaluation.generators import coco_squared
from sklearn.metrics import roc_auc_score
import joblib
from asn.resnets.resnet_asn import ResnetBuilder
from keras.layers import Dense
from keras.models import Model
from asn.training.callbacks import LossHistory_binary
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns

def insert_layer_factory():
    return ASNTransfer_arousal(loc='out')

train = True
test = True
new_tracker = True
filename = 'Results/tracker_ObjectRecognition_8options.pkl'
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
                    '/mnt/Googleplex2/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/train2017_multi_radial.pickle')
    dataset_test = load_pickle(
                    '/mnt/Googleplex2/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/val2017_single_radial.pickle')
    gen_test = coco_squared(dataset_test, batch_size=len(dataset_test['x_ids']) - 1, img_dir=img_dir_test)

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
    gen_train = coco_squared(dataset_train, batch_size=batch_size, img_dir=img_dir_train)
    gen = coco_squared(dataset_val, batch_size=len(dataset_val['x_ids']) - 1, img_dir=img_dir_train)
    (x_val, y_val) = next(gen)

    #  Load the pre-trained model
    modelBase = ResnetBuilder.build_resnet_18_v0((224, 224, 3), 1000)
    modelBase.summary()
    weight_dir = '/mnt/Googleplex2/PycharmProjects/ASN_mac/Resnet18_imagenet/Training/LISA_GPUs/Results/'

    weight_path_resnet = weight_dir + 'weights_training1/FinalWeights.h5'
    modelBase.load_weights(weight_path_resnet)

    temp = Dense(8, activation='sigmoid')(modelBase.layers[-2].output)

    modelFT = Model(inputs=modelBase.input, outputs=temp)

    for l in range(len(modelFT.layers)):
        if modelFT.layers[l].name.startswith('dense'):
            modelFT.layers[l].trainable = True
        else:
            modelFT.layers[l].trainable = False

    del modelBase
    weight_path_new = 'Results/' + dataset

    multi_gpu = False

    learning_mode = 'constant'
    optimizer = 'Adam'
    optimizer_func = eval('optimizers.' + optimizer)
    lr = 1e-4

    modelFT.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                            metrics=['binary_accuracy'])

    if train == True:

        epochs = 75
        checkpointer = ModelCheckpoint(filepath=weight_path_new + '.h5', monitor='val_loss', verbose=1,
                                       save_best_only=True)
        history_call = LossHistory_binary()

        logger = CSVLogger('Results/' + dataset + '.csv')

        callbacks = [checkpointer, history_call, logger]

        history = modelFT.fit_generator(gen_train, steps_per_epoch =len(dataset_train['x_ids'])//batch_size, epochs=epochs,
                                 validation_data=(x_val, y_val), shuffle=True,
                                 callbacks=callbacks)

        tracker[dataset]['Training'] = history.history
        joblib.dump(tracker, filename, compress=True)

        del x_val, y_val

    if test == True:
        (x_test, y_test) = next(gen_test)
        modelFT.load_weights(weight_path_new + '.h5')
        modelArousal = insert_layer_nonseq(modelFT, 'asn_transfer_', insert_layer_factory,
                                           insert_layer_name='Arousal', position='replace',
                                           additionalInput=True)
        modelArousal.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                        metrics=['binary_accuracy'])

        gains = np.round(np.arange(0.7, 1.51, 0.02), 2)

        if new_tracker == True:
            tracker['gains'] = gains

        evaluateTestData = True

        if evaluateTestData == True:
            loss = np.zeros(len(gains))
            acc = np.zeros(len(gains))
            auc = np.zeros(len(gains))
            tracker[dataset]['Test'] = {}
            for g, gain in enumerate(gains):
                print('Gain: ' + str(gain))
                modelArousal.layers[2].set_weights(np.array([[gain]]))
                loss[g], acc[g] = modelArousal.evaluate(x_test, y_test, verbose=False)
                auc[g] = roc_auc_score(y_test, modelArousal.predict(x_test, verbose=False))
                print('AUC: ' + str(auc))

            tracker[dataset]['Test']['binaryCE'] = loss
            tracker[dataset]['Test']['binaryAcc'] = acc
            tracker[dataset]['Test']['AUC'] = auc

            joblib.dump(tracker, filename, compress=True)

        evaluateValData = False

        if evaluateValData == True:
            loss = np.zeros(len(gains))
            acc = np.zeros(len(gains))
            auc = np.zeros(len(gains))

            tracker[dataset]['Validation'] = {}
            for g, gain in enumerate(gains):
                print('Gain: ' + str(gain))
                modelArousal.layers[2].set_weights(np.array([[gain]]))
                loss[g], acc[g] = modelArousal.evaluate(x_val, y_val, verbose=False)
                auc[g] = roc_auc_score(y_val, modelArousal.predict(x_val, verbose=False))
                print('AUC: ' + str(auc))
            tracker['Validation']['binaryCE'] = loss
            tracker['Validation']['binaryAcc'] = acc
            tracker[dataset]['Validation']['AUC'] = auc

            joblib.dump(tracker, filename, compress=True)
