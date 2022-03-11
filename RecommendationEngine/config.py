import multiprocessing
from pathlib import Path
import inspect

class Config:
    def __init__(self, abs_root):
        self.USERNAME = "username"
        self.PASSWORD = "password"
        self.DB = "database"
        self.COLLECTION = "users"
        self.HOST = "localhost"

        self.abs_root = abs_root
        self.MAX_ID_IN_DATASET = 999999 # number of users

        # inputs
        self.data_folder = abs_root / "data/"
        self.raw_data_path = self.data_folder / "ratings.dat"
        # training
        self.test_ratio = 0.10  # fraction of data to be used as test set.
        # 1-match_threshold best rated others are selected, plus others with equal score.
        self.match_threshold = 0.85
        self.num_training_examples = (
            200000  # number of training example to learn from with LGBM
        )

        self.d2v_params = {
            "embedding_size": 100,
            "window": 5,
            "min_count": 1,
            "workers": multiprocessing.cpu_count() - 1,
            "num_epochs": 50,
        }
        self.data_folder = self.data_folder / "prod/"
        self.ids_path = self.data_folder / "hex2int-IDs.pkl"


        self.train_data_path = self.data_folder / "matches_train.dat"
        self.test_data_path = self.data_folder / "matches_test.dat"

        self.u2v_train_data_path = self.data_folder / "matches_train_for_u2v.dat"
        self.u2v_test_data_path = self.data_folder / "matches_test_for_u2v.dat"

        self.rated_embeddings_path = self.data_folder / "models/vectors/rated.vectors"
        self.w2v_model_path = self.data_folder / "models/word_2_vec/model"
        self.rater_embeddings_path = (
            self.data_folder / "models/vectors/rater.vectors.npy"
        )

        self.data_dict_path = self.data_folder / "data_dict.pickle"

        self.keras_model_not_trainable = (
            self.data_folder / "models/keras/classifier_not_trainable.h5"
        )
        self.keras_model_trainable = (
            self.data_folder / "models/keras/classifier_trainable.h5"
        )
        self.keras_model_no_pretraining = (
            self.data_folder / "models/keras/classifier_no_pretraining.h5"
        )


project_absolute_root = (
    Path(inspect.getfile(inspect.currentframe())).absolute().parent.parent
)
config = Config(project_absolute_root)