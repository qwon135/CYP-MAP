from rdkit import Chem
import torch
from pp_unimol import predict_som, to_matrix
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import numpy as np

def validation(model, valid_loader, smile2bond_target, dictionary, loss_fn, n_classes):
    model.eval()    
    valid_loss = 0
    mol_f1s, mol_rec, mol_prc, mol_auc = 0, 0, 0, 0
    bond_f1s, bond_prc, bond_rec, bond_auc = 0, 0, 0, 0
    jac_score = 0

    y_prob_bond, y_true_bond, y_pred_bond = [], [], []
    for batch in valid_loader:
        smile = batch['target']['smi_name'][0]
        mol = Chem.MolFromSmiles(smile)
        som_target = smile2bond_target[ batch['target']['smi_name'][0]]
        som_target = torch.from_numpy(np.array(som_target))
        token_index = torch.BoolTensor( [i not in dictionary.special_index() for i in  batch['net_input']['src_tokens'][0].tolist()])
        
        with torch.no_grad():
            logits, padding_mask = predict_som(model, batch, token_index)
            loss = loss_fn(logits, som_target)
        valid_loss += loss.cpu().item()

        y_true = som_target.tolist()

        y_pred = logits.argmax(-1).cpu().tolist()
        # print(len(set(y_pred)))
        y_prob = torch.nn.functional.softmax( logits, 1 ).tolist()

        y_true_bond += y_true
        y_prob_bond += y_prob
        y_pred_bond += y_pred

        jac_score += jaccard_score(to_matrix(y_true, n_classes), to_matrix(y_pred, n_classes), average='macro')

        mol_f1s += f1_score(y_true, y_pred, average='macro')    
        mol_prc += precision_score(y_true, y_pred, average='macro')
        mol_rec += recall_score(y_true, y_pred, average='macro')
        # mol_auc = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovo')
    valid_loss /= len(valid_loader)

    mol_f1s/= len(valid_loader)
    mol_prc/= len(valid_loader)
    mol_rec/= len(valid_loader)  
    # mol_auc/= len(valid_loader)  

    jac_score /= len(valid_loader)

    y_true_bond = np.array(y_true_bond)
    y_prob_bond = np.array(y_prob_bond)
        
    bond_f1s = f1_score(y_true_bond, y_pred_bond, average='macro')    
    bond_prc = precision_score(y_true_bond, y_pred_bond, average='macro')
    bond_rec = recall_score(y_true_bond, y_pred_bond, average='macro')    
    try:
        bond_auc = roc_auc_score(y_true_bond, y_prob_bond, average='macro', multi_class='ovo')
    except:        
        bond_auc = 0

    return valid_loss, mol_f1s, mol_prc, mol_rec, bond_f1s, bond_prc, bond_rec, bond_auc, jac_score


def get_loaders(task, cyp, class_type):
    task.load_dataset(f'train_{cyp}_class_type{class_type}')
    train_dataset = task.dataset(f'train_{cyp}_class_type{class_type}')
    train_loader = task.get_batch_iterator(train_dataset, batch_size=1, ).next_epoch_itr(shuffle=True)

    task.load_dataset(f'valid_{cyp}_class_type{class_type}')
    valid_dataset = task.dataset(f'valid_{cyp}_class_type{class_type}')
    valid_loader = task.get_batch_iterator(valid_dataset, batch_size=1, ).next_epoch_itr(shuffle=False)

    task.load_dataset(f'test_{cyp}_class_type{class_type}')
    test_dataset = task.dataset(f'test_{cyp}_class_type{class_type}')
    test_loader = task.get_batch_iterator(test_dataset, batch_size=1, ).next_epoch_itr(shuffle=False)
    # test_loader = None
    return train_loader, valid_loader, test_loader