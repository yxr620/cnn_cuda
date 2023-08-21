# 该代码提供训练 TabNet 的工具

# 预计用时 2 小时 (CPU)
#         1 小时 (GPU) 
from tabnet import StackedTabNetRegressor
from scipy.stats.mstats import rankdata
# 请提供以下变量
from setting import FEATURES, PATH_TRAIN_DATA, TARGET_COL, TabNet_MODEL_TARGET_PATH,\
    PATH_TO_SPLITS, RANDOM_SEEDS
import os, re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from model.util import ModelEvaluationICIR, IC
import warnings
warnings.filterwarnings("ignore")
num_rolling = int(len(list(os.listdir(PATH_TO_SPLITS))) / 2)


class TabNet(object):

    def __init__(self):
        self.models = list()
        self.target_path = TabNet_MODEL_TARGET_PATH
        self.random_seeds = RANDOM_SEEDS
        self.splits_path = PATH_TO_SPLITS
        self.train_path = PATH_TRAIN_DATA
        self.feature_name = FEATURES
        self.target_name = TARGET_COL

    def init_models(self):
        for number in tqdm(self.random_seeds):
            print(f"---{number} model init --")
            tmp_model = self.base_model
            self.models.append(tmp_model)

    @property
    def base_model(self):
        base_model = StackedTabNetRegressor(feature_columns=None,
                                            num_regressors=1,
                                            num_layers=2,
                                            num_features=84,
                                            feature_dim=17,
                                            output_dim=16,
                                            num_decision_steps=1)
        return base_model

    @staticmethod
    def get_optimizer(length):
        schedule = tf.keras.optimizers.schedules.CosineDecay(1e-2,
                                                             length // 1024 * 15,
                                                             alpha=1e-2)
        optimizer = tfa.optimizers.AdamW(learning_rate=schedule, weight_decay=lambda: None)
        # tf.keras.optimizers.experimental.AdamW
        # optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=schedule, weight_decay=lambda: None)
        optimizer.weight_decay = lambda: schedule(optimizer.iterations) * 0.1
        return optimizer

    def train(self, trade_date):
        train_dates_path = f"{self.splits_path}/train_{trade_date}.pkl"
        validate_dates_path = f"{self.splits_path}/validate_{trade_date}.pkl"
        if not os.path.exists(train_dates_path):
            raise ValueError(f"{train_dates_path} is not available, please update the train data!")

        dates = pd.read_pickle(train_dates_path) + pd.read_pickle(validate_dates_path)
        train_dates = dates[:-10]
        test_dates = dates[-10:]
        x_test, y_test = [], []
        x_train, y_train = [], []
        for date in tqdm(train_dates):
            try:
                df = pd.read_feather(f"{self.train_path}/{date}.feather").dropna()

                feature_name = FEATURES   # [i for i in df.columns if i.split("_")[0] in self.feature_name]
                x_train.append(np.asarray(df[feature_name]))
                y_train.append(rankdata(df[self.target_name]) / len(df))
            except Exception as error:
                print(error)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        date_len = []
        for date in tqdm(test_dates):
            try:
                df = pd.read_feather(f"{self.train_path}/{date}.feather").dropna()
                feature_name = FEATURES
                x_test.append(np.asarray(df[feature_name]))
                y_test.append(rankdata(df[self.target_name]) / len(df))
                date_len.append(len(y_test))
            except Exception as error:
                print(error)

        x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
        for j in tqdm(range(len(self.random_seeds)), "tabnet training"):
            tf.random.set_seed(self.random_seeds[j])
            tmp_model = self.models[j]
            path = f"{self.target_path}/time_{trade_date}_iter_{j}.h5"
            model_name = f"tabnet_{j}_{test_dates[0]}_{test_dates[-1]}"
            callbacks = [ModelEvaluationICIR(x_test, y_test, date_len, model_name, path, "tmp")]
            optimizer = self.get_optimizer(len(y_train))
            tmp_model.compile(optimizer=optimizer, loss=IC)
            tmp_model.fit(x_train, y_train, batch_size=500, epochs=30, verbose=False, callbacks=callbacks)

        # j = 0
        # for seed in tqdm(self.random_seeds):
        #     tf.random.set_seed(seed)
        #     tmp_model = self.models[j]
        #
        #     optimizer = self.get_optimizer(len(y_train))
        #     tmp_model.compile(optimizer=optimizer, loss=IC)
        #     # tmp_model.fit(x_train, y_train, batch_size=1024, epochs=9, verbose=False)
        #     # tmp_model.save_weights(f"{self.target_path}/time_{max_model_num}_iter_{j}.h5")
        #     # j += 1
        #     tmp_model.fit(x_train, y_train, batch_size=1024, epochs=20, verbose=False)
        #     tmp_model.save_weights(f"{self.target_path}/time_{max_model_num}_iter_{j}.h5")
        #     j += 1
        #     print(f"Model {max_model_num}.{j} Complete")


if __name__ == "__main__":

    # model = TabNet()
    # model.init_models()
    # model.train(31)
    model = TabNet()
    model.init_models()
    model.train("2023-08-02")
