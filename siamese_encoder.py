import numpy as np
import tensorflow as tf
from hyperparameters import num_classes, num_epochs, batch_size, learning_rate, patience, margin
from tensorflow.keras import layers, optimizers, utils, callbacks, Model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils import label_map, contrastive_pair_generator, euclidean_distance


feat_temp_t = np.load('/home/lasii/Research/MM_EXT/NEW/t_a_1d.npy')
feat_spect_t = np.load('/home/lasii/Research/MM_EXT/NEW/t_a_2d.npy')
feat_temp_d = np.load('/home/lasii/Research/MM_EXT/NEW/d_a_1d.npy')
feat_spect_d = np.load('/home/lasii/Research/MM_EXT/NEW/d_a_2d.npy')

train = np.concatenate((feat_temp_t,feat_spect_t),axis = 1)
deploy = np.concatenate((feat_temp_d,feat_spect_d),axis = 1)

train_y, deploy_y = label_map()

pairs_train, labels_train = contrastive_pair_generator(train, train_y, num_classes)
pairs_deploy, labels_deploy = contrastive_pair_generator(deploy, deploy_y, num_classes)


input = layers.Input((train.shape[1]))
x = layers.Flatten()(input)
x = layers.BatchNormalization(epsilon=0.001)(x)
x = layers.Dense(64, activation='selu')(x)

x = layers.Dense(64, activation='selu')(x)
# x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="selu", name = 'feat_layer')(x)
# x = layers.BatchNormalization(epsilon=0.001)(x)
x = layers.Dense(num_classes,activation='softmax')(x)
embedding_network = Model(input, x)
embedding_network.summary()

input_1 = layers.Input((train.shape[1]))
input_2 = layers.Input((train.shape[1]))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="tanh")(normal_layer)
siamese = Model(inputs=[input_1, input_2], outputs=output_layer)


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss


siamese.compile(loss=loss(margin=margin), optimizer= optimizers.Nadam(lr=learning_rate), metrics=["accuracy"])

siamese.summary()
earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

history = siamese.fit(
    [pairs_train[:, 0], pairs_train[:, 1]],
    labels_train,
    callbacks=[earlystop_callback],
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data = ([pairs_deploy[:, 0], pairs_deploy[:, 1]], labels_deploy)
    )

acc_sia = siamese.evaluate([pairs_deploy[:, 0], pairs_deploy[:, 1]], labels_deploy)

##################################################################################################################

feat_ext = Model(embedding_network.input, embedding_network.get_layer('feat_layer').output)

feat_t = feat_ext.predict(train)
feat_d = feat_ext.predict(deploy)


np.save('/home/lasii/Research/MM_EXT/NEW/t_sia_af.npy', feat_t, fix_imports=True)
np.save('/home/lasii/Research/MM_EXT/NEW/d_sia_af.npy', feat_d, fix_imports=True)

##################################################################################################################


nb_classes = 8
num_epochs = 200
batch_size = 32
patience = 50


input = layers.Input((64))
x = layers.Flatten()(input)
x = layers.Dense(1024, activation='selu')(x)
x = layers.Dense(1024, activation='selu')(x)

# x = layers.BatchNormalization(epsilon=0.001)(x)

# x = layers.BatchNormalization(epsilon=0.001)(x)
# x = layers.LayerNormalization(epsilon=0.001)(x)

x = layers.Dense(nb_classes,activation='softmax')(x)
model = Model(input, x)
model.compile(loss='categorical_crossentropy', optimizer = optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])
model.summary()


earlystop_callback_clf = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)



model.fit(
    feat_t,
    utils.to_categorical(train_y, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle=True,
    callbacks=[earlystop_callback_clf],
    validation_data = (feat_d,utils.to_categorical(deploy_y, num_classes=nb_classes)))

score_val = []

 
acc = model.evaluate(feat_d, utils.to_categorical(deploy_y, num_classes=nb_classes))[1]


prf = precision_recall_fscore_support(deploy_y,  np.argmax(model.predict(feat_d),axis = 1), average='macro')
cm = confusion_matrix(deploy_y, np.argmax(model.predict(feat_d),axis = 1))
np.savetxt('sia_af.csv', score_val ,  delimiter=',', fmt ='% s')


