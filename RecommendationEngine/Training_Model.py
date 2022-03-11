from u2v_recommender import *
from config import config
from keras.callbacks import EarlyStopping
from keras.metrics import AUC
from keras import Model, Input
from keras.layers import Embedding, concatenate, Dense
import pandas as pd
import numpy as np
from tqdm import tqdm


def train_model():
    recommender = U2V_Recommender()
    recommender.load_rater_vec(config.rater_embeddings_path)
    recommender.load_rated_vec(config.rated_embeddings_path)

    train = pd.read_csv(config.train_data_path).sample(frac=1)

    x_train = train.iloc[:, :2].values
    y_train = train.iloc[:, 2].values

    max_rater_idx, max_rated_idx, _ = train.max()
    offset_vector = np.zeros((1, config.d2v_params["embedding_size"]))
    rater_embedding_matrix = np.vstack(
        [offset_vector, np.stack(recommender.mean_embeddings.values[:, 0])])

    rated_embedding_matrix = np.zeros(
        (int(max_rated_idx) + 1, config.d2v_params["embedding_size"]))

    for user_id_str in tqdm(recommender.wv.key_to_index.keys()):
        embedding_vector = recommender.wv[user_id_str]
        if embedding_vector is not None:
            user_id_int = int(user_id_str)
            rated_embedding_matrix[user_id_int] = embedding_vector
    # rater
    input_1 = Input(shape=(1,))
    emb_1 = Embedding(
        int(max_rater_idx) + 1,
        config.d2v_params["embedding_size"],
        weights=[rater_embedding_matrix],
        trainable=False,
        input_length=1,
    )
    emb_1 = emb_1(input_1)

    # rated
    input_2 = Input(shape=(1,))
    emb_2 = Embedding(
        int(max_rated_idx) + 1,
        config.d2v_params["embedding_size"],
        weights=[rated_embedding_matrix],
        trainable=False,
        input_length=1,
    )
    emb_2 = emb_2(input_2)

    # dense = Dense
    merge = concatenate([emb_1, emb_2])
    dense1 = Dense(50, activation='relu')(merge)
    dense2 = Dense(25, activation='relu')(dense1)
    dense3 = Dense(1, activation="sigmoid")(dense2)

    model = Model(inputs=[input_1, input_2], outputs=dense3)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', AUC()])

    subset = len(x_train) + 1
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, verbose=2,
        mode='auto', baseline=None, restore_best_weights=True
    )
    history = model.fit([x_train[:subset, 0], x_train[:subset, 1]],
                        y_train[:subset],
                        validation_split=0.1,
                        epochs=500,
                        batch_size=64,
                        callbacks=[early_stopping])

    model.save(config.keras_model_not_trainable)
