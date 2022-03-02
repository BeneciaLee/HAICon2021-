import sys

from pathlib import Path
from datetime import timedelta

import gc
import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from TaPR_pkg import etapr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

torch.manual_seed(428)
torch.cuda.manual_seed_all(428)

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
scaler = pipeline


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

# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84']
# drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63', 'C69', 'C82', 'C85',
#                 'C04','C08', 'C17', 'C20', 'C32', 'C34', 'C40','C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79', 'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80', 'C03', 'C86']
drop_columns = ['C02', 'C09', 'C10', 'C18', 'C19', 'C22', 'C26', 'C29', 'C36', 'C38', 'C39', 'C49', 'C52', 'C55', 'C63',
                'C69', 'C82', 'C85',
                'C04', 'C08', 'C17', 'C20', 'C32', 'C34', 'C40', 'C42', 'C45', 'C46', 'C47', 'C48', 'C61', 'C64', 'C79',
                'C84', 'C05', 'C07', 'C13', 'C23', 'C25', 'C53', 'C60', 'C65', 'C72', 'C80']


def init_data(n_rows=250000):
    TRAIN_DATASET, TEST_DATASET, VALIDATION_DATASET = find_file_name()
    TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
    TRAIN_DF_RAW = TRAIN_DF_RAW.drop(columns=['index'] + drop_columns)
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

WINDOW_SIZE = 31
WINDOW_GIVEN = WINDOW_SIZE - 1
WINDOW_SIZES = [31]
N_HIDDENS = 200
N_LAYERS = 3
BATCH_SIZE = 1800
EPOCH = 500
lr = 0.001
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


class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        # n_tags는 데이터 안에 있는 특징들을 말한다.
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)
        torch.nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        # print("Original Raw Dataset : ", x.shape)
        self.rnn.flatten_parameters()
        x = x.to(DEVICE)
        outs, _ = self.rnn(x)
        # print("outs.shape : ", outs.shape)
        # print(f"What? {len(outs[-1])}")
        outs = F.relu(outs[-1])
        out = self.fc(outs)
        # print("fc.outs.shape : ",out.shape)
        # print("x[0].shape : ",x[0].shape)
        # print("x[0] + out : ",(x[0] + out).shape)
        return x[0] + out


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
    name = f"WINDOWSIZE_{WINDOW_SIZE}_EPOCH_{epoch}_{num}.png"
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
    check_graph(ANOMALY_SCORE, CHECK_ATT, epoch, piece=2, THRESHOLD=THRESHOLD, num=1)
    check_graph(ANOMALY_SCORE, CHECK_ATT, epoch, piece=10, THRESHOLD=THRESHOLD, num=2)
    gc.enable()
    del HAI_DATASET_VALIDATION, CHECK_TS, CHECK_DIST, CHECK_ATT
    gc.collect()

def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
        if e > 200:
            lr_scheduler.step()

        # print(f"{e} Epoch : {epoch_loss}")
        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1

        if e % 100 == 0 and e != 0:
            show_and_save_wave(epoch=e)
            file_name = f"{WINDOW_SIZE}_model_{best['epoch']}_epoch_{best['loss']}_loss.pt"
            with open(file_name, "wb") as f:
                torch.save(
                    {
                        "state": best["state"],
                        "best_epoch": best["epoch"],
                        "loss_history": loss_history,
                    },
                    f,
                )
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
BEST = {}
compare = 0
ch = 0

for window in WINDOW_SIZES:
    WINDOW_SIZE = window
    WINDOW_GIVEN = WINDOW_SIZE - 1
    HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=1)

    MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
    MODEL.to(DEVICE)

    MODEL.train()
    BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, EPOCH)
    print("=" * 40)
    print(f"WINDOW_SIZE : {window}, BEST_LOSS : {BEST_MODEL['loss']}, BEST_EPOCH : {BEST_MODEL['epoch']}")
    print("=" * 40)
    ##########################
    show_and_save_wave(EPOCH)
    ###########################

    file_name = f"{window}_model.pt"
    with open(file_name, "wb") as f:
        torch.save(
            {
                "state": BEST_MODEL["state"],
                "best_epoch": BEST_MODEL["epoch"],
                "loss_history": LOSS_HISTORY,
            },
            f,
        )
    gc.enable()
    del LOSS_HISTORY, HAI_DATASET_TRAIN
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

with open("모델/30결과/31_model.pt", "rb") as f:
    SAVED_MODEL = torch.load(f)

MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
MODEL.cuda()
MODEL.load_state_dict(SAVED_MODEL['state'])

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
ANOMALY_SCORE = remove_valley_noise(ANOMALY_SCORE, cut=0.0115)
ANOMALY_SCORE = remove_blink_noise(ANOMALY_SCORE, limit=40)

check_graph(ANOMALY_SCORE, CHECK_ATT, piece=10, THRESHOLD=0.05)

LABELS = put_labels(ANOMALY_SCORE, 0.05)
FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(TEST_DF_RAW[TIMESTAMP_FIELD]))

submission = pd.read_csv('data/sample_submission.csv')
submission.index = submission['timestamp']
submission.loc[TEST_DF_RAW[TIMESTAMP_FIELD], 'attack'] = FINAL_LABELS
submission.to_csv('31model_0.0115.csv', index=False)
