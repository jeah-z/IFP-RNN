import numpy as np
from ddc_pub import ddc_v3 as ddc
import molvecgen
import rdkit

import h5py, ast, pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
from rdkit.Chem.Draw import IPythonConsole
#We suppres stdout from invalid smiles and validations
from rdkit import rdBase
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"



model_name = "./models/pcb_model"
model = ddc.DDC(model_name=model_name)

# model.gpu()


rdBase.DisableLog ( 'rdApp.*')
def get_descriptors(mol):
    logp  = Descriptors.MolLogP(mol)
    tpsa  = Descriptors.TPSA(mol)
    molwt = Descriptors.ExactMolWt(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    qed   = QED.qed(mol)
     
                     
    # Calculate fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048)
    ecfp4 = np.zeros((2048,))
    DataStructs.ConvertToNumpyArray(fp, ecfp4) 
    a_p = 8e-6
    # Predict activity and pick only the second component
    #active_probability = qsar_model.predict_proba([ecfp4])[0][1]
    return [logp, tpsa, molwt, qed, hba, hbd, a_p]
mol = Chem.MolFromSmiles("c1cccc(OC(=O)C)c1C(=O)O")

# display(mol)
#rdkit.Chem.Draw.ShowMol(mol)
conditions = get_descriptors(mol)
print(conditions)
target = np.array(conditions)

smi = []
for i in range(3000):
    try:
        smiles_out, _ = model.predict(latent=target, temp=1)
        smi.append([smiles_out])
        print(smiles_out)
    except:
        print("oops")
smi_pd = pd.DataFrame(smi, columns=['SMILES'])
smi_pd.to_csv("DDC_output.csv", index=None)