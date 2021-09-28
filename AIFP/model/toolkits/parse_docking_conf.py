import os
import sys

try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob


def parse_vina_conf(vina_conf):
    original_path = os.getcwd()

    protein_file = ""
    ligand_file = ""
    out = ""

    try:
        config_read = open(vina_conf, 'r')
        conf_path = os.path.abspath(vina_conf)
        os.chdir(os.path.dirname(conf_path))
    except FileNotFoundError:
        print("The VINA config file: '%s' can not be found" % (vina_conf))
        sys.exit(1)

    config_lines = [line for line in config_read]
    for line in config_lines:
        uncommented = line.split('#')[0]
        line_list = uncommented.split()

        if not line_list:
            continue
        option = line_list[0]

        if option == "receptor":
            protein_file = line_list[2]
        if option == "ligand":
            ligand_file = line_list[2]
            out = ligand_file[:-6] + "_out" + ligand_file[-6:]
        if option == "out":
            out = line_list[2]

    scorelist = []

    try:
        ligand_out_lines = open(out, 'r')
    except FileNotFoundError:
        print("Ligand output file: '%s' can not be found" % (out))
        sys.exit(1)

    for line in ligand_out_lines:
        line = line.split()
        if len(line) > 2:
            if line[2] == "RESULT:":
                scorelist.append(line[3])

    convert = ob.OBConversion()
    convert.SetInFormat("pdbqt")

    protein = ob.OBMol()
    ligands = ob.OBMol()

    convert.ReadFile(protein, protein_file)

    docked_ligands = []
    docked_proteins = []

    not_at_end = convert.ReadFile(ligands, out)
    while not_at_end:
        docked_ligands.append(ligands)
        ligands = ob.OBMol()
        not_at_end = convert.Read(ligands)

    mollist = []
    ligand_name = ligand_file[:-6]
    for name in range(len(docked_ligands)):
        ligand_pose = ligand_name + '_' + str(name + 1)
        mollist.append(ligand_pose)

    docking_results = {
        'protein': protein,
        'docked_ligands': docked_ligands,
        'scorelist': scorelist,
    }

    os.chdir(original_path)
    return docking_results
