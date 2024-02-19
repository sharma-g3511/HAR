import numpy as np
from tensorflow.keras import layers, optimizers, utils, callbacks, Model
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import argparse
from utils import label_map
from hyperparameters import num_classes, num_epochs, batch_size, learning_rate, patience

parser = argparse.ArgumentParser()

parser.add_argument('--feat_temp_t', type=str, help='train_temporal_feat')
parser.add_argument('--feat_spect_t', type=str, help='train_spectral_feat',)
parser.add_argument('--feat_temp_d', type=str, help='deploy_temporal_feat')
parser.add_argument('--feat_spect_d', type=str, help='deploy_spectral_feat')

args = parser.parse_args()

train_y, deploy_y = label_map()

###############################################################################################

def split_tensor(X):
  s1, s2 = tf.split(X, num_or_size_splits=2,axis = -1)
  return s1,s2

def mul_sca(x):
    return tf.multiply(x[0],x[1])


left_branch_input = layers.Input(shape=(256), name='Left_input')
left_branch_output = layers.Dense(64, activation='selu', name='Left_input_1')(left_branch_input)


right_branch_input = layers.Input(shape=(256), name='Right_input')
right_branch_output = layers.Dense(64, activation='selu')(right_branch_input)

left_nor_output = layers.LayerNormalization()(left_branch_output)
right_nor_output = layers.LayerNormalization()(right_branch_output)


#Merged Layer
merged = layers.concatenate([left_branch_output, right_branch_output], name='Normal_Concatenate')
merge_dense = layers.Dense(64, activation='selu')(merged)
softmax_output = layers.Dense(2, activation='softmax', name='soft_score')(merge_dense)

weight_left, weight_right = layers.Lambda(split_tensor)(softmax_output)

score_left_mul = layers.Lambda(mul_sca)([left_nor_output, weight_left])

score_right_mul = layers.Lambda(mul_sca)([right_nor_output, weight_right])

additive_score = layers.Add()([score_left_mul, score_right_mul])

feat_layer = layers.Dense(64, activation="selu", name = 'feat_layer')(additive_score)

final_model_output = layers.Dense(8, activation='sigmoid')(feat_layer)

###############################################################################################


model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,name='Final_output')

model.compile(loss='categorical_crossentropy', optimizer = optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])

model.summary()

earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

model.fit(
    [np.load(args.feat_temp_t), np.load(args.feat_spect_t)],
    utils.to_categorical(train_y, num_classes=num_classes),
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle = True,
    callbacks=[earlystop_callback],
    validation_data = ([np.load(args.feat_temp_d), np.load(args.feat_spect_d)],utils.to_categorical(deploy_y, num_classes=num_classes)))

###############################################################################################

score_val = []

acc = model.evaluate([np.load(args.feat_temp_d), np.load(args.feat_spect_d)],utils.to_categorical(deploy_y, num_classes=num_classes))[1]
prf = precision_recall_fscore_support(deploy_y,  np.argmax(model.predict([np.load(args.feat_temp_d), np.load(args.feat_spect_d)]),axis = 1), average='macro')
cm = confusion_matrix(deploy_y,  np.argmax(model.predict([np.load(args.feat_temp_d), np.load(args.feat_spect_d)]),axis = 1))
score_val.append([acc, prf[0], prf[1], prf[2], cm])
# np.savetxt('af_12_.csv', score_val ,  delimiter=',', fmt ='% s')

# feature_extractor = Model(inputs=model.inputs,outputs=model.get_layer(name="feat_layer").output)
# feat_t = feature_extractor.predict([np.load(args.feat_temp_t), np.load(args.feat_spect_t)])
# feat_d = feature_extractor.predict([np.load(args.feat_temp_d), np.load(args.feat_spect_d)])

# np.save('/home/lasii/Research/MM_EXT/NEW/t_af_.npy', feat_t, fix_imports=True)
# np.save('/home/lasii/Research/MM_EXT/NEW/d_af_.npy', feat_d, fix_imports=True)


