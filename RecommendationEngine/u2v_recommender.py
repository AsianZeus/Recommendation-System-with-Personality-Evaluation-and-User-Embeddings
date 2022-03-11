import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing
import pickle

np.random.seed(0)
tqdm.pandas()


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        if self.epoch == 1:
            print(f"Epoch: {self.epoch}    Loss: {loss}")
        else:
            print(
                f"Epoch: {self.epoch}    Loss: {loss - self.loss_previous_step}"
            )
        self.epoch += 1
        self.loss_previous_step = loss


class ModelSaver(CallbackAny2Vec):
    def __init__(
        self, u2v_object, rated_embeddings_path, w2v_model_path, log_frequency=5
    ):
        self.epoch = 1
        self.log_frequency = log_frequency
        self.u2v_object = u2v_object
        self.rated_embeddings_path = rated_embeddings_path
        self.w2v_model_path = w2v_model_path

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        if self.epoch % self.log_frequency == 0:
            self.u2v_object.w2v_model = model
            self.u2v_object.wv = model.wv
            self.u2v_object.save_rated_vec(self.rated_embeddings_path)
            self.u2v_object.save_w2v_model(self.w2v_model_path)
        self.epoch += 1


class U2V_Recommender:
    def __init__(
        self,
        embedding_size=100,
        window=3,
        min_count=1,
        workers=multiprocessing.cpu_count() - 1,
        num_epochs=50,
        sample=0,
    ):
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.num_epochs = num_epochs
        self.sample = sample

        self.wv = None
        self.mean_embeddings = (
            None
        )
        self.data_dict = None

    def fit_rated_embeddings(
        self, u2v_train, w2v_model_path, rated_embeddings_path, resume_training=False):
        u2v_train_iterator = self.build_data_iterator(u2v_train)

        loss_logger = LossLogger()
        model_saver = ModelSaver(self, rated_embeddings_path, w2v_model_path)
        self.w2v_model = Word2Vec(
            vector_size=self.embedding_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sample=self.sample,
            sg=1,
            hs=0,
            negative=5,
            callbacks=[loss_logger, model_saver],
            seed=0,
        )
        model = self.w2v_model
        if resume_training:
            model = self.load_w2v_model(w2v_model_path)
            model.build_vocab(u2v_train_iterator, update=True)
        elif model.train_count == 0:
            model.build_vocab(u2v_train_iterator)
        model.train(
            u2v_train_iterator,
            total_examples=model.corpus_count,
            epochs=self.num_epochs,
            compute_loss=True,
        )

        self.w2v_model = model
        self.wv = model.wv
        self.save_rated_vec(rated_embeddings_path)
        self.save_w2v_model(w2v_model_path)

    def fit_rater_embeddings(self, input_train, save_path=False):
        A_col, B_col, m_col = input_train.columns
        train_ = input_train.copy()
        train_ = train_[train_[m_col] > 0]
        train_[B_col] = train_[B_col].apply(self.get_single_rated_vec)
        train_ = train_.dropna(
            subset=[B_col]
        )
        train_ = train_.groupby(A_col)[B_col].apply(np.mean)
        train_.index = train_.index.astype(str)
        train_ = train_.to_frame()
        self.mean_embeddings = train_
        if save_path:
            self.save_rater_vec(save_path)

    def build_data_iterator(self, data):
        class shuffle_generator:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                self.data.apply(np.random.shuffle)
                return shuffle_generator_iter(self.data)

        class shuffle_generator_iter:
            def __init__(self, data):
                self.i = 0
                self.data = data
                self.data_length = len(data)

            def __iter__(self):
                return self

            def __next__(self):
                if self.i < 5 * self.data_length:
                    if self.i % self.data_length == 0:
                        self.data.apply(np.random.shuffle)
                    i = self.i
                    self.i += 1
                    return self.data[i % self.data_length]
                else:
                    raise StopIteration()

        return shuffle_generator(data)

    def get_single_rated_vec(self, rated_id):
        try:
            return self.wv[str(rated_id)]
        except KeyError:
            return None

    def get_single_rater_vec(self, rater_id):
        try:
            return self.mean_embeddings.loc[str(rater_id)].values
        except KeyError:
            return None

    def save_rated_vec(self, wordvectors_path):
        wordvectors_path.parent.mkdir(parents=True, exist_ok=True)
        self.wv.save(str(wordvectors_path))

    def load_rated_vec(self, wordvectors_path):
        self.wv = KeyedVectors.load(str(wordvectors_path), mmap="r")
        return self.wv

    def save_w2v_model(self, w2v_model_path):
        w2v_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.w2v_model.save(str(w2v_model_path))

    def load_w2v_model(self, w2v_model_path):
        self.w2v_model = Word2Vec.load(str(w2v_model_path), mmap="r")
        return self.w2v_model

    def save_rater_vec(self, rater_embeddings_path):
        rater_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(rater_embeddings_path,
                arr=self.mean_embeddings.reset_index().values)

    def load_rater_vec(self, rater_embeddings_path):
        arr = np.load(rater_embeddings_path, allow_pickle=True)
        self.mean_embeddings = pd.DataFrame(
            index=arr[:, 0].astype(str), data=arr[:, 1])
        return self.mean_embeddings

    def save_data_dict(self, data_dict_path):
        with open(data_dict_path, "wb") as handle:
            pickle.dump(self.data_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def load_data_dict(self, data_dict_path):
        with open(data_dict_path, "rb") as handle:
            self.data_dict = pickle.load(handle)
        return self.data_dict
