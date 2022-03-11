import ast
from u2v_recommender import U2V_Recommender
import pandas as pd
from config import config

def load_u2v_formated_data(data_path):
    df = pd.read_csv(data_path)
    df = df["rated"].map(ast.literal_eval)
    return df

def train_embeddings():
    recommender = U2V_Recommender(**config.u2v_params)
    train = pd.read_csv(config.train_data_path)
    u2v_train = load_u2v_formated_data(config.u2v_train_data_path)
    resume_training = False
    recommender.fit_rated_embeddings(
        u2v_train,
        config.w2v_model_path,
        config.rated_embeddings_path,
        resume_training=resume_training,
    )
    del u2v_train
    recommender.fit_rater_embeddings(
        train, save_path=config.rater_embeddings_path)
