import numpy as np
from keras.models import model_from_json, Model
from keras.layers import Dense
from asn.arousal.layers import ASNTransfer_arousal
from asn.arousal.utils import insert_layer_nonseq, makeImageNoImageDataset
from asn.utils import load_pickle
from asn.evaluation.generators import coco_squared
import joblib
from sklearn.linear_model import LogisticRegression
from asn.resnets.resnet_asn import ResnetBuilder

def insert_layer_factory():
    return ASNTransfer_arousal(loc='out')


multi_gpu = False
train = False
recheck = True

np.random.seed(3)
# for dataset in ['1_street','2_food']:
dataset = '1_street'

# Create a simple task: distinguish images from non-images
img_dir_train = '/mnt/Googleplex2/coco/images/' + dataset + '/train2017_multi_radial/'
img_dir_test = '/mnt/Googleplex2/coco/images/' + dataset + '/val2017_single_radial/'

dataset_base = load_pickle(
    '/mnt/Googleplex2/PycharmProjects/ASN_mac/COCO/Dataset/' + dataset + '/train2017_multi_radial.pickle')

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
gen_val = coco_squared(dataset_val, batch_size=len(dataset_val['x_ids']), img_dir=img_dir_train)

chunks = 10  # for loading
dataset_train = {}
dataset_train['x_ids'] = dataset_base2['x_ids'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['y'] = dataset_base2['y'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['dataType'] = 'train2017'
gen_train = coco_squared(dataset_train, batch_size=len(dataset_train['x_ids']) // chunks, img_dir=img_dir_train)

del dataset_base, dataset_base2

mode = 'random'
num_rands = [1.125, 1.25, 1.5, 2, 3, 4, 6, 20]
new_tracker = False
filename = 'Results/tracker_LayerInfo.pkl'
if new_tracker == True:
    tracker = {}
    tracker['num_rands'] = num_rands
else:
    tracker = joblib.load(filename)

for num_rand in num_rands:
    print(num_rand)
    if mode == 'gray':
        tag = mode
    else:
        tag = mode + str(num_rand)
    if tag not in tracker:
        tracker[tag] = {}

    learning_mode = 'constant'
    optimizer = 'Adam'
    optimizer_func = eval('optimizers.' + optimizer)
    lr = 1e-4

    # Test the weights with varying levels of gain.

    gains = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04,
             1.06, 1.08, 1.1, 1.12, 1.14, 1.16, 1.18, 1.2, 1.22, 1.24, 1.26, 1.28, 1.3, 1.32, 1.34, 1.36, 1.38, 1.4]

    layers = ['Arousal_18', 'Arousal_17', 'Arousal_16', 'Arousal_15', 'Arousal_14', 'Arousal_13', 'Arousal_12',
              'Arousal_11', 'Arousal_10', 'Arousal_9', 'Arousal_8', 'Arousal_7', 'Arousal_6', 'Arousal_5',
              'Arousal_4', 'Arousal_3', 'Arousal_2', 'Arousal_1', 'Arousal_0']
    layers = list(reversed(layers))
    trainings = 10
    layers_i = [3, 5, 7, 9, 11, 13, 15, 18]

    if new_tracker == True:
        tracker['gains'] = gains
        tracker['layers'] = layers
        tracker['training'] = trainings

    evaluateTestData = True

    if 'training_' + str(trainings-1) not in tracker[tag]:
        (x_test, y_test) = next(gen_test)
        x_test_binary, y_test_binary = makeImageNoImageDataset(x_test, mode=mode, num_rand=num_rand)
        del x_test, y_test

    if evaluateTestData == True:
        images = 1000
        # scores = np.zeros((iterations, len(layers), len(gains)))
        for i in range(trainings):
            if 'training_' + str(i) not in tracker[tag]:
                (x_test, y_test) = next(gen_test)
                x_test_binary, y_test_binary = makeImageNoImageDataset(x_test, mode=mode, num_rand=num_rand)
                del x_test, y_test

                modelBase = ResnetBuilder.build_resnet_18_v0((224, 224, 3), 1000)
                modelBase.summary()

                temp = Dense(1, activation='sigmoid')(modelBase.layers[-2].output)

                modelBinary = Model(inputs=modelBase.input, outputs=temp)

                del modelBase

                weight_path_new = 'Results/BinaryEvolvingTask_' + tag + '_' + str(i)
                # reload the best weights from the training:
                modelBinary.load_weights(weight_path_new + '.h5')

                modelArousal = insert_layer_nonseq(modelBinary, 'asn_transfer_', insert_layer_factory,
                                                   insert_layer_name='Arousal', position='replace',
                                                   additionalInput=True)

                modelArousal.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr),
                                     metrics=['binary_accuracy'])

                print('Training :' + str(i))
                scores = np.zeros((len(layers), len(gains)))
                # make a random index
                idx = np.arange(len(x_test_binary))
                np.random.shuffle(idx)

                feature_images = x_test_binary[idx[:images]]
                feature_labels = y_test_binary[idx[:images]]

                test_images = x_test_binary[idx[-images:]]
                test_labels = y_test_binary[idx[-images:]]

                for l in layers_i:
                    print(layers[l])
                    modelInspect = Model(inputs=modelArousal.input, outputs=modelArousal.get_layer(layers[l]).output)

                    for g, gain in enumerate(gains):
                        print(gain)
                        # tracker[tag][layer][gain] = {}
                        modelInspect.layers[2].set_weights(np.array([[gain]]))
                        # obtain predictions
                        preds = modelInspect.predict(feature_images)

                        clf = LogisticRegression(random_state=i).fit(preds.reshape([images, -1]), feature_labels)

                        preds_test = modelInspect.predict(test_images)
                        scores[l, g] = clf.score(preds_test.reshape([images, -1]), test_labels)
                        print(scores[l, :])

                    tracker[tag]['training_' + str(i)] = scores

                    joblib.dump(tracker, filename, compress=True)

    if 'x_test_binary' in locals():
        del x_test_binary, y_test_binary, modelArousal, modelBinary
