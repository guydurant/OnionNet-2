from rdkit import Chem
from generate_features import generate_features, keys
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
outermost = 0.05 * (shells+1)
ncutoffs = np.linspace(0.1, outermost, shells)

def make_complex_pdb(key, protein_file, ligand_file):
    # ligand_file = ligand_file.replace('_pymol.sdf', '.mol2')
    lig_mol = Chem.MolFromMolFile(ligand_file)
    if lig_mol is None:
        print(f'Could not load {ligand_file}')
    lig_pdb = Chem.MolToPDBBlock(lig_mol)
    lig_pdb = lig_pdb.replace('UNL', 'LIG')
    with open(protein_file, 'r') as f:
        prot_pdb = f.read()
    with open(f'temp_files/{key}_complex.pdb', 'w') as f:
        f.write(prot_pdb)
        f.write(lig_pdb)
    return None

def generate_features_per_complex(key, protein_file, ligand_file):
    make_complex_pdb(key, protein_file, ligand_file)
    features = generate_features(f'temp_files/{key}_complex.pdb', ncutoffs)
    os.system(f'rm temp_files/{key}_complex.pdb')
    return features

def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [os.path.join(data_dir, protein_file) for protein_file in df['protein']]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df['ligand']]
    keys = df['key']
    pks = df['pk']
    return protein_files, ligand_files, keys, pks

def generate_all_features(csv_file, data_dir):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    all_features = Parallel(n_jobs=-1)(delayed(generate_features_per_complex)(key, protein_file, ligand_file) for key, protein_file, ligand_file in tqdm(zip(keys, protein_files, ligand_files)))
    # for protein_file, ligand_file, key, in tqdm(zip(protein_files, ligand_files, keys)):
    #     # try:
    #         features = generate_features_per_complex(key, protein_file, ligand_file)
    #         all_features.append(features)
    #     # except:
        #     print(f'Could not generate features for {key}')

    all_features = np.array(all_features)  
    all_elements = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
    all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
    feat_keys = ["_".join(x) for x in list(itertools.product(all_residues, all_elements))]
    columns = []
    for i, n in enumerate(feat_keys * len(ncutoffs)):
        columns.append(f'{n}_{i}')
    print(all_features.shape)
    print(len(columns))
    return pd.DataFrame(all_features, columns=columns)

def train_model(args, model_name):
    model = create_model(args.shape, args.rate, args.clipvalue, lr=args.lr)
    train = pd.read_csv(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv', index_col=0)
    val = pd.read_csv(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv', index_col=0)
    n_features = 21*8*62
    X_train = train.values[:, :n_features]
    X_valid = val.values[:, :n_features]
    scaler = preprocessing.StandardScaler()
    X_train_std = scaler.fit_transform(X_train).reshape([-1] + args.shape)
    X_valid_std = scaler.transform(X_valid).reshape([-1] + args.shape)
    joblib.dump(scaler, f'temp_models/{model_name}.scaler')
    y_train = pd.read_csv(args.csv_file)['pk'].values
    y_valid = pd.read_csv(args.val_csv_file)['pk'].values
    print(y_valid)
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=args.patience,
                                            verbose=1, mode='auto', )
    logger = tf.keras.callbacks.CSVLogger("logfile", separator=',', append=False)
    bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=f'temp_models/{model_name}', verbose=1, save_best_only=True)
        
    history = model.fit(X_train_std, y_train,
                        validation_data = (X_valid_std, y_valid),   
                        epochs = args.epochs,
                        batch_size = args.batchsz,
                        verbose=1,
                        callbacks=[stop, logger, bestmodel])
                    
def predict(args):
    alpha = args.alpha

    test = pd.read_csv(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv', index_col=0)
    test_index = pd.read_csv(args.val_csv_file)['key'].values
    true_values = pd.read_csv(args.val_csv_file)['pk'].values
    
    X_test = test.values
    
    scaler = joblib.load(f'temp_models/{args.model_name}.scaler')
    X_test_std = scaler.transform(X_test).reshape([-1] + args.shape)
    
    model = tf.keras.models.load_model(f'temp_models/{args.model_name}',
            custom_objects={'RMSE': RMSE,
                'PCC': PCC,
                'PCC_RMSE': PCC_RMSE})

    pred_pKa = model.predict(X_test_std).ravel()
    return pd.DataFrame({'key': test_index, 'pred': pred_pKa, 'pk': true_values})



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='train.csv')
    parser.add_argument('--val_csv_file', type=str, default='val.csv')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--val_data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='test')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument("--shape", type=int, default=[84, 124, 1], nargs="+",
                        help="Input. Reshape the features.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Input. The learning rate.")
    parser.add_argument("--batchsz", type=int, default=64,
                        help="Input. The number of samples processed per batch.")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Input. The dropout rate.")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Input. The alpha value in loss function.")
    parser.add_argument("--clipvalue", type=float, default=0.01,
                        help="Input. The threshold for gradient clipping.")
    parser.add_argument("--n_features", type=int, default=10416,
                        help="Input. The number of features for each complex. \n"
                             "When shells N=62, n_feautes=21*8*62.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Input. The number of times all samples in the training set pass the CNN model.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Input. Number of epochs with no improvement after which training will be stopped.")
    args = parser.parse_args()
    print(args.rate)
    if args.train:
        if not os.path.exists(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv'):
            df = generate_all_features(args.csv_file, args.data_dir)
            df.to_csv(f'temp_features/{args.csv_file.split("/")[-1].split(".")[0]}_features.csv')
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'):
            df = generate_all_features(args.val_csv_file, args.val_data_dir)
            df.to_csv(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv')
        train_model(args, args.model_name)
    if args.predict:
        if not os.path.exists(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv'):
            df = generate_all_features(args.val_csv_file, args.val_data_dir)
            df.to_csv(f'temp_features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.csv')
        df = predict(args)
        df.to_csv(f'results/{args.model_name}_{args.val_csv_file.split("/")[-1]}', index=False)


