import numpy as np
from utils import heart_data, label_map
from hyperparameters import num_classes, num_epochs, batch_size, learning_rate, patience
from tensorflow.keras import layers, optimizers, utils, callbacks, Model
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

train, deploy = heart_data()
train_y, deploy_y = label_map()

###############################################################################################

input = layers.Input((train.shape[1],train.shape[2]))
x = layers.Conv1D(32,1,activation='selu',padding='same', kernel_initializer='he_normal')(input)
x = layers.MaxPooling1D(pool_size=2,strides=1,padding='same')(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization(epsilon=0.001)(x)
x = layers.Dense(64, activation='selu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation="selu", name = 'feat_layer')(x)
x = layers.BatchNormalization(epsilon=0.001)(x)
x = layers.Dense(num_classes, activation='softmax')(x)
model = Model(input, x)
model.compile(loss='categorical_crossentropy', optimizer = optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])
model.summary()


score_val = []


earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

model.fit(
    train,
    utils.to_categorical(train_y, num_classes=num_classes),
    batch_size=batch_size,
    epochs=num_epochs,
    shuffle = True,
    callbacks=[earlystop_callback],
    validation_data = (deploy,utils.to_categorical(deploy_y, num_classes=num_classes)))

acc = model.evaluate(deploy,utils.to_categorical(deploy_y, num_classes=num_classes))[1]
prf = precision_recall_fscore_support(deploy_y,  np.argmax(model.predict(deploy),axis = 1), average='macro')
cm = confusion_matrix(deploy_y,  np.argmax(model.predict(deploy),axis = 1))
score_val.append([acc, prf[0], prf[1], prf[2], cm])
# np.savetxt('h_1d.csv', score_val ,  delimiter=',', fmt ='% s')

feature_extractor = Model(inputs=model.inputs,outputs=model.get_layer(name="feat_layer").output)

feat_t = feature_extractor.predict(train)
feat_d = feature_extractor.predict(deploy)