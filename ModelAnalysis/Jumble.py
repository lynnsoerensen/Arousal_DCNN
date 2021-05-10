import numpy as np
from keras.models import model_from_json, Model
from keras.layers import Dense
from asn.arousal.layers import ASNTransfer_arousal, spatialJumble
from asn.arousal.utils import insert_layer_nonseq, makeImageNoImageDataset
from asn.utils import load_pickle
from asn.evaluation.generators import coco_squared
import joblib
from keras import optimizers
from asn.resnets.resnet_asn import ResnetBuilder


def insert_layer_factory():
    return ASNTransfer_arousal(loc='out')

multi_gpu = False
train = False

np.random.seed(3)

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

chunks = 10 # for loading
dataset_train= {}
dataset_train['x_ids'] = dataset_base2['x_ids'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['y'] = dataset_base2['y'][int(np.ceil(len(dataset_base2['x_ids']) * proportion_val)):]
dataset_train['dataType'] = 'train2017'
gen_train = coco_squared(dataset_train, batch_size=len(dataset_train['x_ids'])//chunks, img_dir=img_dir_train)

del dataset_base, dataset_base2

mode = 'random'
num_rands = [1.25, 3, 20]
trainings = 2
places = ['batch_normalization_4', 'batch_normalization_6', 'batch_normalization_8', 'batch_normalization_10',
          'batch_normalization_12', 'batch_normalization_14', 'batch_normalization_16', 'batch_normalization_18']

rates = np.arange(0, 1.1, 0.1)[1:]
iterations = 10
learning_mode = 'constant'
optimizer = 'Adam'
optimizer_func = eval('optimizers.' + optimizer)
lr = 1e-4

new_tracker = False
filename = 'Results/tracker_spatialJumble.pkl'
if new_tracker == True:
    tracker = {}
    tracker['places'] = places
    tracker['rates'] = rates
    tracker['iterations'] = iterations

else:
    tracker = joblib.load(filename)


for num_rand in num_rands:
    print(num_rand)
    if mode == 'gray':
        tag = mode
    elif mode == '2_food':
        tag = mode
    else:
        tag = mode + str(num_rand)
    if tag not in tracker:
        tracker[tag] = {}

    for t in range(trainings):
        if 'training_' + str(t) not in tracker[tag]:
            tracker[tag]['training_' + str(t)] = {}


            modelBase = ResnetBuilder.build_resnet_18_v0((224, 224, 3), 1000)
            modelBase.summary()

            temp = Dense(1, activation='sigmoid')(modelBase.layers[-2].output)

            modelBinary = Model(inputs=modelBase.input, outputs=temp)

            del modelBase

            weight_path_new = 'Results/BinaryEvolvingTask_' + tag + '_' + str(t)

            # reload the best weights from the training:
            modelBinary.load_weights(weight_path_new + '.h5')

            for l in range(len(modelBinary.layers)):
                if modelBinary.layers[l].name.startswith('dense'):
                    pass
                else:
                    modelBinary.layers[l].trainable = False

            modelBinary.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr), metrics=['binary_accuracy'])


            (x_test, y_test) = next(gen_test)
            x_test_binary, y_test_binary = makeImageNoImageDataset(x_test, mode=mode, num_rand=num_rand)
            del x_test, y_test

            # Test the weights with varying levels of gain.
            tracker[tag]['training_' + str(t)]['baseline'] =  modelBinary.evaluate(x_test_binary, y_test_binary)

            for b in range(len(places)):
                if places[b] not in tracker[tag]['training_' + str(t)]:
                    scores = np.zeros((len(rates), iterations, 2))
                    for r in range(len(rates)):

                        def insert_jumble():
                            return spatialJumble(rate=rates[r])


                        modelJumble = insert_layer_nonseq(modelBinary, places[b], insert_jumble,
                                                                insert_layer_name='Jumble', position='after')

                        modelJumble.compile(loss='binary_crossentropy', optimizer=optimizer_func(lr=lr), metrics=['binary_accuracy'])

                        for i in range(iterations):
                            scores[r,i, :] = modelJumble.evaluate(x_test_binary, y_test_binary)
                        print('rate: ' + str(rates[r]))
                        print(scores[r, :, 1])


                    tracker[tag]['training_' + str(t)][places[b]] = scores

                    joblib.dump(tracker, filename, compress=True)

    if 'x_test_binary' in locals():
        del x_test_binary, y_test_binary, modelJumble, modelBinary


