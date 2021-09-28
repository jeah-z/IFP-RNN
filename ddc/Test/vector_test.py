from vectorizers import SmilesVectorizer


charset = '@C)(=cOn1S2/H[N]\\'
smilesvec = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=120,
                charset=charset,
                binary=False,
            )

import rdkit 
smi="c1cccc(OC(=O)C)c1C(=O)O"
mol=[]
mol.append(rdkit.Chem.MolFromSmiles(smi))
mol.append(rdkit.Chem.MolFromSmiles(smi))
onehot=smilesvec.transform(mol)
# print(onehot)
# onehot.shape
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
print(char_to_int)
print(int_to_char)