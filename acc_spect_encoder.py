import numpy as np
from utils import acceleration_data, label_map,stft_generator
from hyperparameters import num_classes, num_epochs, batch_size, learning_rate, patience, chans
from tensorflow.keras import layers, optimizers, backend, utils, callbacks, Model, models
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

train, deploy = acceleration_data()
train_y, deploy_y = label_map()


stft_train = stft_generator(train)
stft_deploy = stft_generator(deploy)

###############################################################################################

def model_a(row,col):
    model_a = models.Sequential()
    model_a.add(layers.Conv2D(64, (3, 3), activation='selu', input_shape=(chans,row,col), data_format = 'channels_first'))
    model_a.add(layers.Conv2D(64, (3, 3), activation='selu'))
    model_a.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model_a.add(layers.Flatten()) 
    model_a.add(layers.Dropout(0.5))
    model_a.add(layers.Dense(256, activation='selu', name = 'feat_layer'))
    model_a.add(layers.Dropout(0.5))
    model_a.add(layers.Dense(num_classes, activation='softmax'))
    model_a.compile(loss='categorical_crossentropy', optimizer = optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])
    model_a.summary()
    return model_a

score_val = []
backend.clear_session()
model = model_a(stft_train.shape[2],stft_train.shape[3])
earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

model.fit(
    stft_train,
    utils.to_categorical(train_y, num_classes=num_classes),
    batch_size=batch_size,
    epochs=num_epochs,
    callbacks=[earlystop_callback],
    validation_data = (stft_deploy, utils.to_categorical(deploy_y, num_classes=num_classes)))

###############################################################################################

acc = model.evaluate(stft_deploy,utils.to_categorical(deploy_y, num_classes=num_classes))[1]
prf = precision_recall_fscore_support(deploy_y,  np.argmax(model.predict(stft_deploy),axis = 1), average='macro')
cm = confusion_matrix(deploy_y,  np.argmax(model.predict(stft_deploy),axis = 1))
score_val.append([acc, prf[0], prf[1], prf[2], cm])
np.savetxt('a_2d.csv', score_val ,  delimiter=',', fmt ='% s')


feature_extractor = Model(inputs=model.inputs,outputs=model.get_layer(name="feat_layer").output)

feat_t = feature_extractor.predict(stft_train)
feat_d = feature_extractor.predict(stft_deploy)


    
    
    
