import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, utils, callbacks
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils import acceleration_data, label_map
from hyperparameters import num_classes, num_epochs, batch_size, learning_rate, patience



train, deploy = acceleration_data()
train_y, deploy_y = label_map()

train = train.reshape(train.shape[0],1,train.shape[1])
deploy = deploy.reshape(deploy.shape[0],1,deploy.shape[1])

###############################################################################################




def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="selu")(res)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="selu")(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="selu")(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="selu")(x)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="selu")(x)

    # x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="selu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


input_shape = train.shape[1:]

model = build_model(
    input_shape,
    head_size=16,
    num_heads=2,
    ff_dim=32,
    num_transformer_blocks=1,
    mlp_units=[64],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Nadam(lr=learning_rate), metrics=['accuracy'])
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
# np.savetxt('STFT_transformer.csv', score_val ,  delimiter=',', fmt ='% s')
