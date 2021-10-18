# IFP-RNN
A molecule generative model used interaction fingerprint (docking pose) as constraints.
# Install

```
conda create -n IFP-RNN python=3.6
conda install openbabel -c openbabel
conda install cudatoolkit=10.1
pip install -r requirement.txt
```

# Docking
## Small dataset less than 100, 000
First a working directory needs to be created in the root directory to save the docking results. For example, "Test/Data" directory is created in the IFP-RNN project. The follow docking command is executed in the Data sub-directory. Though both glide and vina can be used in this project, only open-source vina is used in this example.
The docking command is as below!
```
python ../../AIFP/dock_batch.py --n_jobs 20 --machine 188 --save_path ./test_results --dataset ../../Dataset/test.csv --config_vina ./config_vina.txt --mgltools ../../MGLTools-1.5.7
```
- --n_jobs  This option determines how many jobs will run in parallel. Keep in mind that each job take 4 cpu cores in default.
- --machine It is optional, which will remind which machine is using if you have multiple machines working in the same directory.
- --save_path Folder to save the docking results
- --dataset The dataset of ligands. CSV file is necessary, and two columns will be read, namely "ChEMBL ID": name of ligands,"Smiles": SMILES of the ligands. Please check the test.csv file in the Dataset as a reference to create a file of your project.
- --config_vina The config of docking with vina. Please check the config_vina.txt in the Test/Data folder while creating your version of docking config.
- --mgl_tools The MGLTools to carried out docking. It can be download from http://mgltools.scripps.edu/downloads
## Large dataset larger than 100, 000
If a huge dataset is used for docking like ChEMBL27.csv, another option --subset should be used to avoid too many docking results (*_out.pdbqt) are saving in the same directory.
```
python ../../AIFP/dock_batch.py --n_jobs 20 --machine 188 --save_path ./test_results --dataset ../../Dataset/test.csv --config_vina ./config_vina.txt --mgltools ../../MGLTools-1.5.7 --subset 0
 ```
 - --subset This option is for choosing a part of ligand in the dataset for docking. 0 means the first 100, 000 ligands; 1 means the second 100, 000 ligands and so on...(0,1,2,3,4...) The results will be saved in the sub-directiory as subset option. And the script can be run in parallel for the subsets.

## Results
After docking a set of results (*_out.pdbqt) are obtained, which is stored in the --save_path.

# Calculate interaction fingerprint (IFP)
## Create IFP reference
The purpose of this section is to detect the atom or residue of the receptor pocket which can form five major interactions, namely H-bond, Halogen-bond, Aromatic interaction, Electrostatic interaction, and Hydrophobic interaction with ligands. The order of reference atoms and residues will determine the order of IFP bits. And each reference atom and residude has five bits that record the existence of five types of interactions. The bits with particular order as the reference is the IFP to encode the docking pose used in this project.
We have finished docking. Let's enter the root work directory, 'Test', and construct the IFP. First let's construct the IFP reference that is a list of atoms and residue of the receptor, which can formed interactions with the ligands used to created the reference.
```
python ../AIFP/create_reference.py --config ./config_ifp.txt
```
- --config  The config file to create IFP including a set of parameter. Users should write the file carefully. An example file is config_ifp.txt in the 'Test' folder. 

## Results
Two files, namely 'refer_atoms_list.pkl', 'refer_res_list.pkl', were obtained in the ./obj folder. They store the reference atoms and residue list separately.

## Construct interaction fingerprint (IFP) based on the reference
This section will construct the IFP based the docking results and IFP reference.
```
python ../AIFP/create_IFP_batch.py --config ./config_ifp.txt --n_jobs 50 --save test_ifp
```
- --n_jobs Cpu cores will be used.
- --save The name of IFP file.
## Results
Two files, namely 'test_ifp_AAIFP.csv',  'test_ifp_ResIFP.csv' were obtained in the working folder. They store the atom-based IFP (AIFP) and residue-based IFP, separately.

# Prepare input for constrained molecule generative model (cRNN).
After we obtain IFP, some additional work needs to be done before feed the data into the cRNN model. A script, 'get_smi_score.py', has been written to carry out this task. 
## For AIFP model
```
python ../AIFP/get_smi_score.py --path ./Data/test_results  --dataset ./test_ifp_AAIFP.csv --smi ../Dataset/test.csv  --model aifp
```
- path The path that stores the docking results. If there is sub-folder, such as '0', '1', it is necessary to add the sub-folder './Data/test_results/0'. The script will create an input file for each sub-folder, and the training script will combine them together.
- --dataset The IFP file to be perpared.
- --smi The file of liagands dataset used for docking.
- --model The model type that the script is preparing for. There are mainly three types of model, namely 'aifp','dScorePP','ecfp'.
## For dScorePP+AIFP model
```
python ../AIFP/get_smi_score.py --path ./Data/test_results  --dataset ./test_ifp_AAIFP.csv --smi ../Dataset/test.csv  --model dScorePP
```
## For ECFP+AIFP model
```
python ../AIFP/get_smi_score.py --path ./Data/test_results  --dataset ./test_ifp_AAIFP_AIFPsmi.csv --smi ../Dataset/test.csv  --model ecfp
```
It is needed to notice that the dataset of AIFP model is used to create the ECFP model input for convenience.
## If you want try different number of poses included for the same ligand.
```
python ../AIFP/pose_select.py --input test_ifp_AAIFP_AIFPsmi.csv,test_ifp_AAIFP_dScorePP.csv,test_ifp_AAIFP_ecfpSmi.csv,test_ifp_ResIFP_AIFPsmi.csv --max_idx 1

```
- --input The inputs of cRNN model. Multiple files separated with a comma can be processed in one command line.
- --max_idx The number of poses you want to include in the cRNN training, five poses will be used by default.
## For Residue-based model
ResIFP shares the same preparation code above as the AIFP.

## Results
After preparation of the cRNN input, three types, namely, '*_AIFPsmi.csv', '*_dScorePP.csv', '*_ecfpSmi.csv'  of file will be obtained for each type of model, separately.

# Directly calculate interaction fingerprint (IFP) with SDF file 
If the users don't want to follow the docking rutine of this project, and want to calculate the IFP from the docking results in SDF format. The jobs can be done followint the steps bellow.
## SDF format rules
The SDF format should include three propertis for each molecules, namely 'Docking_score', 'Pose_id', 'SMILES'. If you conducted docking with glide, and have tranformed the docking results into SDF format. Further preparation can be done with the following command.
```
python ../AIFP/prepare_sdf_glide.py --smi /data/ranting/work/tbk1/x01d/x01 --sdf /data/ranting/work/tbk1/x01d/sp_4euu_min_x01-2_pv.sdf --work_dir ./

```
- --sdf The docking result of glide.
- --smi The SMILES of the molecules, which is optional. If it is not provided or it is invalid the SMILES will be genereated from the mol file instead.
- work_dir The directory to save the results.

## Create IFP reference (SDF)
```
python ../AIFP/create_reference_sdf.py --config config_ifp_sdf.txt --protein 4euu_cpx_optim_pro.pdb --sdf ./test.sdf --n_jobs 50
```
- --protein The protein should be in pdb format.
A folder of 'obj' will be created, which includes three files, namely, 'protein.pdbqt',  'refer_atoms_list.pkl',  'refer_res_list.pkl'. To be noted, the files will be reused in the background. So your working directory should include the folder all the time.

## Construct interaction fingerprint (SDF)
```
python ../AIFP/create_IFP_sdf.py --config config_ifp_sdf.txt  --sdf ./test.sdf --n_jobs 50
```
It is needed to be noted, an 'info.csv' file, including Ligand name, docking score, pose id, SMILES etc., will be generated in the working directory, and will be used later.s

# Prepare input for constrained molecule generative model (SDF).

```
python ../AIFP/prepare_ddc_input_sdf.py --dataset IFP_AAIFP.csv --info Tmp_test/info.csv --type ecfp
```


# Training
Train command of cRNN model is as below. If you want to change the default training parameters, please edit the 'train_ddc.py' file directly.
```
python -u ../train_ddc.py --train_csv ./AIFP_files/cdk2_chembl0_ResIFP_AIFPsmi_5pose.csv,./AIFP_files/cdk2_chembl1_ResIFP_AIFPsmi_5pose.csv,./AIFP_files/cdk2_chembl2_ResIFP_AIFPsmi_5pose.csv,./AIFP_files/cdk2_crystal_ResIFP_AIFPsmi_5pose.csv,./AIFP_files/cdk2_active_ResIFP_AIFPsmi_5pose.csv  --load_pkl 0  --save cdk2_Res_AIFPsmi_train/5pose/cdk2_res_AIFPsmi_5pose
```
- --train_csv The multiple input files prepared in the previous section. The input files should be separated with comma.
- --load_pkl This option dertermint if loading the data from pickle file directly. This option should be '0', if the training is run first time. And after processing the inputs will be saved in a pickle file and can be reused next time after setting this option to '1'.
- --save This option determine where to save the checkpoint of trained models during the training process.ss sss