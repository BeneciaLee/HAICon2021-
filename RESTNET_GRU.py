import sys

from pathlib import Path
from datetime import timedelta

import gc
import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from TaPR_pkg import etapr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
TAG_MIN = 0
TAG_MAX = 0
TRAIN_DATASET, TEST_DATASET, VALIDATION_DATASET = 0, 0, 0

from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

# configure our pipeline
pipeline = Pipeline([('normalizer', StandardScaler()),
                     ('scaler', MinMaxScaler())])
# pipeline = Pipeline([('scaler', MinMaxScaler())])
scaler = pipeline

import os
import random


def seed_everything(seed=428):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets]).reset_index()


def normalize(df, TAG_MIN, TAG_MAX):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf


def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))


def find_file_name():
    TRAIN_DATASET = sorted([x for x in Path("data/train/").glob("*.csv")])
    TEST_DATASET = sorted([x for x in Path("data/test/").glob("*.csv")])
    VALIDATION_DATASET = sorted([x for x in Path("data/validation/").glob("*.csv")])
    return TRAIN_DATASET, TEST_DATASET, VALIDATION_DATASET


TRAIN_DATASET, TEST_DATASET, VALIDATION_DATASET = find_file_name()
TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
TRAIN_DF_RAW = TRAIN_DF_RAW.drop(columns=['index'])

# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C116', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C116', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63',
#                 'C69', 'C116', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34',
#                 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13',
#                 'C20']

# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C116', 'C85']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C116', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34', 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13', 'C20']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63',
#                 'C69', 'C116', 'C85',
#                 'C04', 'C08', 'C17', 'C20', 'C32', 'C34', 'C40', 'C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79',
#                 'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C116', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34', 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13', 'C20']

# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80', 'C03', 'C86']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34', 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13', 'C20'
#                   ,'C84', 'C86', 'C03']


# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80', 'C03', 'C86']

# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34', 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13', 'C20']

#

# drop_columns = ['C01', 'C02', 'C04', 'C05', 'C06', 'C08', 'C09', 'C10', 'C13', 'C14', 'C18', 'C19', 'C21', 'C22', 'C25', 'C26', 'C29', 'C31', 'C33', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C45', 'C46', 'C47', 'C49', 'C52', 'C55', 'C59', 'C60', 'C61', 'C63', 'C64', 'C65', 'C67', 'C69', 'C71', 'C72', 'C74', 'C78', 'C80', 'C82', 'C84', 'C85', 'C86', 'C50', 'C20', 'C48', 'C34', 'C17']

# drop_columns = ['C01', 'C02', 'C04', 'C05', 'C06', 'C08', 'C09', 'C10', 'C13', 'C14', 'C18', 'C19', 'C21', 'C22', 'C25', 'C26', 'C29', 'C31', 'C33', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C45', 'C46', 'C47', 'C49', 'C52', 'C55', 'C59', 'C60', 'C61', 'C63', 'C64', 'C65', 'C67', 'C69', 'C71', 'C72', 'C74', 'C78', 'C80', 'C82', 'C84', 'C85', 'C86', 'C50', 'C20', 'C48']

drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85', 'C01', 'C04', 'C06', 'C25', 'C42', 'C50', 'C65', 'C86', 'C05', 'C08', 'C17', 'C34', 'C40', 'C45', 'C46', 'C47', 'C48', 'C50', 'C61', 'C64', 'C67', 'C71', 'C72', 'C74', 'C80', 'C07', 'C13', 'C20']
# drop_columns += ['C30', 'C31', 'C62', 'C73', 'C76', 'C83', 'C57', 'C59']
# drop_columns += ['C30', 'C31', 'C62', 'C73', 'C76', 'C83', 'C57']
drop_columns += ['C30', 'C31', 'C62', 'C73','C59']

def init_data(n_rows=250000):
    TRAIN_DATASET, TEST_DATASET, VALIDATION_DATASET = find_file_name()
    TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
    TRAIN_DF_RAW = TRAIN_DF_RAW.drop(columns=['index'] + list(set(drop_columns)))
    print(TRAIN_DF_RAW.shape)
    TRAIN_DF_RAW = TRAIN_DF_RAW[:] if n_rows == -1 else TRAIN_DF_RAW[:n_rows]

    VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
    VALID_COLUMNS_IN_TRAIN_DATASET

    TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
    TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

    TRAIN_DF = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]
    VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
    VALIDATION_DF = VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]
    TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
    TEST_DF = TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]

    TRAIN_DF = pd.DataFrame(scaler.fit_transform(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])).ewm(alpha=0.9).mean()

    VALIDATION_DF = pd.DataFrame(scaler.transform(VALIDATION_DF[VALID_COLUMNS_IN_TRAIN_DATASET])).ewm(alpha=0.9).mean()

    TEST_DF = pd.DataFrame(scaler.transform(TEST_DF[VALID_COLUMNS_IN_TRAIN_DATASET])).ewm(alpha=0.9).mean()

    return TRAIN_DF, TRAIN_DF_RAW, VALIDATION_DF, VALIDATION_DF_RAW, VALID_COLUMNS_IN_TRAIN_DATASET, TEST_DF, TEST_DF_RAW


TRAIN_DF, TRAIN_DF_RAW, VALIDATION_DF, VALIDATION_DF_RAW, VALID_COLUMNS_IN_TRAIN_DATASET, TEST_DF, TEST_DF_RAW = init_data(
    n_rows=-1)


WINDOW_SIZE = 61
WINDOW_GIVEN = WINDOW_SIZE - 1
WINDOW_SIZES = [61]
N_HIDDENS = 150
N_LAYERS = 3
BATCH_SIZE = 1800
EPOCH = 551
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=5, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in trange(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                    self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        # print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i: i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential()

        self.elu = nn.ELU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.elu(x)
        return x


class Test(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=68, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(68),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # (BatchSize, Channel, Time)
        #
        self.block1 = BasicBlock(in_channels=68, out_channels=68)
        self.block2 = BasicBlock(in_channels=68, out_channels=68)

        self.block3 = BasicBlock(in_channels=68, out_channels=128, stride=2)
        self.block4 = BasicBlock(in_channels=128, out_channels=128)

        self.block5 = BasicBlock(in_channels=128, out_channels=256, stride=2)
        self.block6 = BasicBlock(in_channels=256, out_channels=256)

        self.block7 = BasicBlock(in_channels=256, out_channels=512, stride=2)
        self.block8 = BasicBlock(in_channels=512, out_channels=512)

        self.input = torch.nn.Linear(512, 36)

        self.rnn = torch.nn.GRU(
            input_size=36,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )

        self.fc = torch.nn.Linear(N_HIDDENS * 2, 36)

    def forward(self, x):
        x = x.to(DEVICE)
        x_zero = x[:, 0, :]
        x = x.transpose(1, 2)  # (batch, timesteps, channel) -> (batch, channel, timesteps)

        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = x.transpose(1, 2)  # (batch, channel, timesteps) -> (batch, timesteps, channel)
        x = self.input(x)
        x = F.elu(x)



        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)


        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        outs = F.elu(outs[-1])
        out = self.fc(outs)

        return x_zero + out

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.997)
    loss_fn = torch.nn.L1Loss()
    epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].to(DEVICE)
            # print(f"\nGiven.shape : {given.shape}")
            guess = model(given)
            # print(f"Guess.shape : {guess.shape}")
            answer = batch["answer"].to(DEVICE)
            # print(f"Answer.shape : {answer.shape}")
            loss = loss_fn(answer, guess)
            # print("Answer : ", answer[0])
            # print("Guess : ", guess[0])
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        # print(f"{e} Epoch : {epoch_loss}")


        if e > 100:
            lr_scheduler.step()
        # lr_scheduler.step()

        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            gc.enable()
            del best
            gc.collect()
            best = {}
            best["model"] = model
            best["loss"] = epoch_loss
            best["epoch"] = e + 1


        if e % 100 == 0 and e != 0:
            show_and_save_wave(epoch = e)
            BEST = best['model']
            file_name = f"test2/{WINDOW_SIZE}_model_{best['epoch']}_epoch_{best['loss']}_loss.pt"
            torch.save(BEST, file_name)
            MODEL.train()

    gc.enable()
    del dataloader
    gc.collect()

    return best, loss_history


################################
# 검증 데이터셋 구축하는 부분입니다.
################################
# inference(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE)
def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(batch_size))

    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )


def check_graph(xs, att,epoch, piece=2, THRESHOLD=None, num=1):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD != None:
            axs[i].axhline(y=THRESHOLD, color='r')
    name = f"test2/WINDOWSIZE_{WINDOW_SIZE}_EPOCH_{epoch}_{num}.png"
    plt.savefig(name, dpi=300)
    plt.show()


def show_and_save_wave(epoch):
    HAI_DATASET_VALIDATION = HaiDataset(
        VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD], stride=1
    )

    MODEL.eval()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, MODEL, BATCH_SIZE)

    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

    THRESHOLD = 0.03
    check_graph(ANOMALY_SCORE, CHECK_ATT,epoch, piece=2, THRESHOLD=THRESHOLD,num=1)
    check_graph(ANOMALY_SCORE, CHECK_ATT, epoch, piece=10, THRESHOLD=THRESHOLD,num=2)
    gc.enable()
    del HAI_DATASET_VALIDATION, CHECK_TS, CHECK_DIST, CHECK_ATT
    gc.collect()

BEST = 0

for window in WINDOW_SIZES:
    HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=1)

    MODEL = Test()
    MODEL.to(DEVICE)

    MODEL.train()
    BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, 500)
    print("=" * 40)
    print(f"WINDOW_SIZE : {window}, BEST_LOSS : {BEST_MODEL['loss']}, BEST_EPOCH : {BEST_MODEL['epoch']}")
    print("=" * 40)
    ##########################
    show_and_save_wave(EPOCH)
    ###########################

    # file_name = f"{window}_model.pt"

    BEST = BEST_MODEL['model']
    file_name = f"{window}_model_{BEST_MODEL['epoch']}_epoch_{BEST_MODEL['loss']}_loss.pt"
    torch.save(BEST, file_name)

    gc.enable()
    del BEST_MODEL, LOSS_HISTORY, HAI_DATASET_TRAIN
    gc.collect()


"""
    위의 부분을 통해서 가장 좋은 모델을 선정한다.
    train이라는 함수에서는 100에폭당 모델을 저장했다. 따라서, 저장된 모델을 비교 분석하여서 가장 좋은 모델을 선택하면된다.
    아래에 있는 부분 부터는 위에서 선택된 가장 좋은 모델을 불러와서 결과를 예측하겠다. 
"""

#================================================================================================================
"""
    아래에 있는 경로는 본인이 저장한 파일 경로를 설정하면 된다.
"""
MODEL = torch.load('test2/61_model_499_epoch_1.4960836723912507_loss.pt')

THRESHOLD = 0.03

def show_and_save_wave():
    HAI_DATASET_VALIDATION = HaiDataset(
        VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD], stride=1
    )

    MODEL.eval()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, MODEL, BATCH_SIZE)

    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

    # check_graph(ANOMALY_SCORE, CHECK_ATT, piece=10, THRESHOLD=THRESHOLD)
    gc.enable()
    del HAI_DATASET_VALIDATION, CHECK_DIST
    gc.collect()
    return ANOMALY_SCORE, CHECK_TS, CHECK_ATT


def check_graph(xs, att, piece=2, THRESHOLD=None):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD != None:
            axs[i].axhline(y=THRESHOLD, color='r')
    name = f"WINDOWSIZE_{WINDOW_SIZE}.png"
    plt.savefig(name, dpi=300)

"""
    결과 예측값에서 발생되는 valley noise를 없애고자 추가한 알고리즘 입니다. 
    이를 통해서 모델에서 발생되는 valley noise를 제거할 수 있습니다.
    결과적으로 Recall 값에서의 성능 향상을 보일 수 있습니다. 
"""
def remove_valley_noise(data, cut):
    temp = data
    gap = 50

    check_index = np.where(temp > cut)
    check_index = check_index[0]
    check_index.sort()

    for i in check_index:
        for j in check_index[1:]:
            if (j - i) < gap:
                temp[i:j + 1] = 0.1

    return temp

"""
    결과 예측값에서 발생되는 blink noise를 없애고자 추가한 알고리즘 입니다. 
    이를 통해서 모델에서 발생되는 blink noise를 제거할 수 있습니다.
    결과적으로 Precision 값에서의 성능 향상을 보일 수 있습니다. 
"""
def remove_blink_noise(ANOMALY_SCORE, limit):
    data = ANOMALY_SCORE.copy()
    index = np.where(data > 0.095)
    index = index[0]
    index.sort()

    print(index)

    cnt = 0
    left = 0
    right = 0
    for i in range(len(index)):
        if cnt == 0:
            left = index[i]

        if i != (len(index) - 1) and (index[i + 1] - index[i]) == 1:
            cnt += 1
            continue
        else:
            right = index[i]
            if left == right:
                data[left] = 0.001
                cnt = 0
                continue

            if cnt < limit:
                data[left:right + 1] = 0.001
                cnt = 0
                continue
            else:
                cnt = 0

    return data

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

HAI_DATASET_TEST = HaiDataset(
    TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None, stride=1
)

MODEL.eval()

CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_TEST, MODEL, BATCH_SIZE)
ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
ANOMALY_SCORE = remove_valley_noise(ANOMALY_SCORE, cut=0.035)
ANOMALY_SCORE = remove_blink_noise(ANOMALY_SCORE, limit=150)

check_graph(ANOMALY_SCORE, CHECK_ATT, piece=10, THRESHOLD=0.05)

LABELS = put_labels(ANOMALY_SCORE, 0.05)
FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(TEST_DF_RAW[TIMESTAMP_FIELD]))

submission = pd.read_csv('data/sample_submission.csv')
submission.index = submission['timestamp']
submission.loc[TEST_DF_RAW[TIMESTAMP_FIELD], 'attack'] = FINAL_LABELS
submission.to_csv('resnet.csv', index=False)




