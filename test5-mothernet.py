from datahandles import FewShotDataset
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from ticl.prediction import MotherNetClassifier
from datetime import datetime


def write_to_file(file_name, row):
    row_str = ", ".join([str(item) for item in row])
    with open(file=file_name, mode='a') as f:
        f.write(f"{row_str}\n")

datasets = ["bank", "blood", "calhousing", "heart", "income"]
n_shotss = [4, 8, 16, 32, 64]
random_seeds = [14, 26, 42, 58, 97]

now = datetime.now()
timestamp_string = now.strftime("%Y%m%d%H%M%S")
out_file_name = f"mothernet/results_{timestamp_string}.csv"
try:
    with open(out_file_name, 'x') as f:
        f.write(f"dataset, n_shots, random_seed, roc_auc, accuracy\n")
    print(f"File {out_file_name} created.")
except FileExistsError:
    print(f"File {out_file_name} already exists.")



for ds_name in datasets:
    print(f"{ds_name} dataset started")
    n_queries_dict = {"bank": 43211, "blood": 374, "calhousing": 19640, "heart": 459, "income": 44222}
    test_ds = FewShotDataset(
        dataset_names=[ds_name],
        data_root="./data",
        split="test",
        split_size=1,
        n_shots=n_queries_dict[ds_name],
        n_queries=n_queries_dict[ds_name],
        queries_same_as_shots=True,
        max_n_features=None,
        balance_labels=False,
        col_permutation=False,
        shuffle=True,
        debug=False,
        random_seed=True,
        shots_with_labels=False
    )
    X_test = test_ds[0]["queries_x"].numpy()
    y_test = test_ds[0]["queries_y"].to(torch.long).numpy()

    for n_shots in n_shotss:
        print(f"{n_shots} shots started")
        for random_seed in random_seeds:
            train_ds = FewShotDataset(
                    dataset_names=[ds_name],
                    data_root="./data",
                    split="train",
                    split_size=1,
                    n_shots=n_shots,
                    n_queries=n_shots,
                    queries_same_as_shots=True,
                    max_n_features=None,
                    balance_labels=True,
                    col_permutation=False,
                    shuffle=True,
                    debug=False,
                    random_seed=random_seed,
                    shots_with_labels=False
                )
            X_train = train_ds[0]["queries_x"].numpy()
            y_train = train_ds[0]["queries_y"].to(torch.long).numpy()

            classifier = MotherNetClassifier(device='cpu')

            classifier.fit(X_train, y_train)
            y_eval = classifier.predict(X_test)
            y_prob = classifier.predict_proba(X_test)

            acc = accuracy_score(y_test, y_eval)
            roc = roc_auc_score(y_test, y_prob[:,1])

            row = [ds_name, n_shots, random_seed, roc, acc]
            write_to_file(out_file_name, row)

        print(f"{n_shots} shots done")
    print(f"{ds_name} dataset done")

print("All done")