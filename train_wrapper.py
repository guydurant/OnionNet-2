import openbabel
from generate_features import generate_features
from train import create_model, PCC_RMSE, PCC, RMSE
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import tensorflow as tf
from joblib import Parallel, delayed
import joblib
import itertools

shells = 62
outermost = 0.05 * (shells + 1)
ncutoffs = np.linspace(0.1, outermost, shells)


def make_complex_pdb(key, protein_file, ligand_file, csv_file):
    if not os.path.exists(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}'):
        os.makedirs(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}')
    # ligand_file = ligand_file.replace('_pymol.sdf', '.mol2')
    # lig_mol = Chem.MolFromMolFile(ligand_file)
    # if lig_mol is None:
    #     print(f"Could not load {ligand_file}")
    # lig_pdb = Chem.MolToPDBBlock(lig_mol)
    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("sdf", "pdb")
    lig_mol = openbabel.OBMol()
    ob_conversion.ReadFile(lig_mol, ligand_file)
    lig_pdb = ob_conversion.WriteString(lig_mol)
    lig_pdb = lig_pdb.split("\n")
    for i in lig_pdb[:]:
        # replace residue name with LIG
        if i.startswith("ATOM") or i.startswith("HETATM"):
            # replace resi name with LIG
            new_i = i[:17] + "LIG" + i[20:]
            lig_pdb[lig_pdb.index(i)] = new_i
        else:
            lig_pdb.remove(i)
    with open(protein_file, "r") as f:
        prot_pdb = f.read()
    prot_pdb_split = prot_pdb.split("\n")
    for p in prot_pdb_split[:]:
        if not p.startswith("ATOM"):
            prot_pdb_split.remove(p)
    with open(
        f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_complex.pdb", "w"
    ) as f:
        for p in prot_pdb_split:
            f.write(p + "\n")
        for l in lig_pdb:
            f.write(l + "\n")
    return None


def generate_features_per_complex(key, protein_file, ligand_file, csv_file):
    make_complex_pdb(key, protein_file, ligand_file, csv_file)
    features = generate_features(
        f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_complex.pdb",
        ncutoffs,
    )
    os.system(
        f"rm data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_complex.pdb"
    )
    return features


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def generate_all_features(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    all_features = Parallel(n_jobs=-1)(
        delayed(generate_features_per_complex)(key, protein_file, ligand_file, csv_file)
        for key, protein_file, ligand_file in tqdm(
            zip(keys, protein_files, ligand_files)
        )
    )
    all_features = np.array(all_features)
    all_elements = ["H", "C", "O", "N", "P", "S", "Hal", "DU"]
    all_residues = [
        "GLY",
        "ALA",
        "VAL",
        "LEU",
        "ILE",
        "PRO",
        "PHE",
        "TYR",
        "TRP",
        "SER",
        "THR",
        "CYS",
        "MET",
        "ASN",
        "GLN",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
        "HIS",
        "OTH",
    ]
    feat_keys = [
        "_".join(x) for x in list(itertools.product(all_residues, all_elements))
    ]
    columns = []
    for i, n in enumerate(feat_keys * len(ncutoffs)):
        columns.append(f"{n}_{i}")
    return pd.DataFrame(all_features, columns=columns)


def train_model(args, model_name):
    print("creating model")
    model = create_model(args.shape, args.rate, args.clipvalue, lr=args.lr)
    full = pd.read_csv(
        f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv',
        index_col=0,
    )
    val = full.sample(1000, random_state=42)
    train = full.drop(val.index)
    n_features = 21 * 8 * 62
    X_train = train.values[:, :n_features]
    X_valid = val.values[:, :n_features]
    scaler = preprocessing.StandardScaler()
    X_train_std = scaler.fit_transform(X_train).reshape([-1] + args.shape)
    X_valid_std = scaler.transform(X_valid).reshape([-1] + args.shape)
    joblib.dump(scaler, f"data/models/{model_name}.scaler")
    all_y = pd.read_csv(args.csv_file)["pk"].values
    y_train = all_y[train.index]
    y_valid = all_y[val.index]
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=args.patience,
        verbose=1,
        mode="auto",
    )
    logger = tf.keras.callbacks.CSVLogger("logfile", separator=",", append=False)
    bestmodel = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"data/models/{model_name}", verbose=1, save_best_only=True
    )
    # run keras model on GPU

    history = model.fit(
        X_train_std,
        y_train,
        validation_data=(X_valid_std, y_valid),
        epochs=args.epochs,
        batch_size=args.batchs,
        verbose=1,
        callbacks=[stop, logger, bestmodel],
    )


def predict(args):
    alpha = args.alpha
    test = pd.read_csv(
        f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv',
        index_col=0,
    )
    test_index = pd.read_csv(args.val_csv_file)["key"].values
    true_values = pd.read_csv(args.val_csv_file)["pk"].values

    X_test = test.values

    scaler = joblib.load(f"data/models/{args.model_name}.scaler")
    X_test_std = scaler.transform(X_test).reshape([-1] + args.shape)

    model = tf.keras.models.load_model(
        f"data/models/{args.model_name}",
        custom_objects={"RMSE": RMSE, "PCC": PCC, "PCC_RMSE": PCC_RMSE},
    )

    pred_pKa = model.predict(X_test_std).ravel()
    return pd.DataFrame({"key": test_index, "pred": pred_pKa, "pk": true_values})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str, default="val.csv")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--shape",
        type=int,
        default=[84, 124, 1],
        nargs="+",
        help="Input. Reshape the features.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Input. The learning rate."
    )
    parser.add_argument(
        "--batchs",
        type=int,
        default=64,
        help="Input. The number of samples processed per batch.",
    )
    parser.add_argument(
        "--rate", type=float, default=0.0, help="Input. The dropout rate."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Input. The alpha value in loss function.",
    )
    parser.add_argument(
        "--clipvalue",
        type=float,
        default=0.01,
        help="Input. The threshold for gradient clipping.",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=10416,
        help="Input. The number of features for each complex. \n"
        "When shells N=62, n_feautes=21*8*62.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Input. The number of times all samples in the training set pass the CNN model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Input. Number of epochs with no improvement after which training will be stopped.",
    )
    args = parser.parse_args()
    if args.train:
        if not os.path.exists("data/models"):
            os.makedirs("data/models")
        if not os.path.exists("data/features"):
            os.makedirs("data/features")
        if not os.path.exists(
            f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'
        ):
            df = generate_all_features(args.csv_file, args.data_dir)
            df.to_csv(
                f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'
            )
        if args.gpus > 0:
            with tf.device("/device:GPU:0"):
                print("GPU Name", tf.config.list_physical_devices("GPU"))
                train_model(args, args.model_name)
        else:
            raise ValueError("Please use a GPU to train the model.")
    if args.predict:
        if not os.path.exists(
            f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'
        ):
            df = generate_all_features(args.val_csv_file, args.val_data_dir)
            df.to_csv(
                f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'
            )
        df = predict(args)
        if not os.path.exists("data/results"):
            os.makedirs("data/results")
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    if not args.train and not args.predict:
        raise ValueError("Please specify --train or --predict or both.")
