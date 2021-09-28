from __future__ import print_function

import sys
import os
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob


def path_to_filename(path_input):
    name = os.path.basename(path_input)
    name = name.replace('_', '.').split('.')
    file_name = name[0]
    return file_name


def parse_protein_vina(protein_file):
    protein_name = path_to_filename(protein_file)
    print(protein_file)
    convert = ob.OBConversion()
    convert.SetInFormat("pdbqt")
    protein = ob.OBMol()
    convert.ReadFile(protein, protein_file)

    docking_results = {
        'name': protein_name,
        'protein': protein
    }
    return docking_results


def parse_ligand_vina(ligand_file):
    ligand_name = path_to_filename(ligand_file)
    # read the scorelist in the pdbqt file
    scorelist = []
    try:
        ligand_out_lines = open(ligand_file, 'r')
    except FileNotFoundError:
        print("Ligand output file: '%s' can not be found" % (out))
        sys.exit(1)
    for line in ligand_out_lines:
        line = line.split()
        if len(line) > 2:
            if line[2] == "RESULT:":
                scorelist.append(line[3])
    # process docking pose of ligands after docking
    convert = ob.OBConversion()
    convert.SetInFormat("pdbqt")
    ligands = ob.OBMol()
    docked_ligands = []
    not_at_end = convert.ReadFile(ligands, ligand_file)
    while not_at_end:
        docked_ligands.append(ligands)
        ligands = ob.OBMol()
        not_at_end = convert.Read(ligands)

    docking_results = {
        'docked_ligands': docked_ligands,
        'scorelist': scorelist
    }
    return docking_results


def parse_config_vina(config_file):

    try:
        configread = open(config_file, 'r')
    except FileNotFoundError:
        print("The config file: '%s' can not be found" % (config))
        sys.exit(1)

    configlines = [line for line in configread]
    configread.close()
    # the setting format in the config as : protein_file   name.pdbqt
    for line in configlines:
        uncommented = line.split('#')[0]
        line_list = uncommented.split()

        if not line_list:
            continue
        option = line_list[0]

        if option == "protein_file":
            protein = line_list[1]

        elif option == "ligand_folder":
            ligand_folder = line_list[1]
        elif option == "refer_ligands_path":
            refer_ligands_folder = line_list[1]

        elif option == "refer_cutoff":
            refer_cutoff = line_list[1]
        elif option == "hbond_cutoff":
            hbond_cutoff = line_list[1]
        elif option == "halogenbond_cutoff":
            halogenbond_cutoff = line_list[1]
        elif option == "electrostatic_cutoff":
            electrostatic_cutoff = line_list[1]
        elif option == "hydrophobic_cutoff":
            hydrophobic_cutoff = line_list[1]
        elif option == "pistack_cutoff":
            pistack_cutoff = line_list[1]
        elif option == "prepare_ligand4":
            prepare_ligand4 = line_list[1]
        elif option == "vina_path":
            vina_path = line_list[1]

    parse_result = {
        'protein': protein,
        'ligand_folder': ligand_folder,
        'refer_ligands_folder': refer_ligands_folder,
        'refer_cutoff': refer_cutoff,
        'hbond_cutoff': hbond_cutoff,
        'halogenbond_cutoff': halogenbond_cutoff,
        'electrostatic_cutoff': electrostatic_cutoff,
        'hydrophobic_cutoff': hydrophobic_cutoff,
        'pistack_cutoff': pistack_cutoff,
        'prepare_ligand4': prepare_ligand4,
        'vina_path': vina_path
    }
    return parse_result


def parse_config():
    #  Default configuration
    config = ''

    #  Identifiers initialization
    docking_method = ""
    docking_conf = ""
    similarity_coef = []
    simplified_ref = []
    full_ref = []
    full_nobb_ref = []

    use_backbone = True
    res_name = []
    res_num = []
    res_weight1 = []
    res_weight2 = []
    res_weight3 = []
    res_weight4 = []
    res_weight5 = []
    docking_score = True

    output_mode = {"full": False, "full_nobb": False, "simplified": False}
    output_mode_undefined = {"full": False,
                             "full_nobb": False, "simplified": False}
    '''
    Default values
    '''

    simplified_outfile = "simplified_ifp.csv"
    full_outfile = "full_ifp.csv"
    full_nobb_outfile = "full_nobb_ifp.csv"
    sim_outfile = "similarity.csv"
    logfile = "hippos.log"

    if len(sys.argv) > 1:
        config = sys.argv[1]
    else:
        print("HIPPOS config file not defined, using config.txt instead")
        print("To change the config file, use it as argument after HIPPOS")
        print("Example:")
        print("\t hippos <config file>\n")

    try:
        configread = open(config, 'r')
    except FileNotFoundError:
        print("The config file: '%s' can not be found" % (config))
        sys.exit(1)

    configlines = [line for line in configread]
    configread.close()

    for line in configlines:
        uncommented = line.split('#')[0]
        line_list = uncommented.split()

        if not line_list:
            continue
        option = line_list[0]

        if option == "docking_method":
            value = line_list[1]
            method = ["vina", "plants"]
            if value in method:
                docking_method = value
            else:
                print("docking method '%s' is not recognized" % (value))
                sys.exit(1)

        elif option == "docking_conf":
            docking_conf = line_list[1]

        elif option == "similarity_coef":
            value = line_list[1:]
            similarity_coef = value

        elif option == "simplified_ref":
            value = line_list[1:]
            simplified_ref = value

        elif option == "full_ref":
            value = line_list[1:]
            full_ref = value

        elif option == "full_nobb_ref":
            value = line_list[1:]
            full_nobb_ref = value

        elif option == "use_backbone":
            value = line_list[1]
            bool_val = ["yes", "no"]
            if value in bool_val:
                if value == "no":
                    use_backbone = False

        elif option == "residue_name":
            res_name = line_list[1:]
        elif option == "residue_number":
            res_num = line_list[1:]

        elif option == "res_weight1":
            res_weight1 = line_list[1:]
        elif option == "res_weight2":
            res_weight2 = line_list[1:]
        elif option == "res_weight3":
            res_weight3 = line_list[1:]
        elif option == "res_weight4":
            res_weight4 = line_list[1:]
        elif option == "res_weight5":
            res_weight5 = line_list[1:]

        elif option == "docking_score":
            value = line_list[1]
            bool_val = ["yes", "no"]
            if value in bool_val:
                if value == "no":
                    docking_score = False

        elif option == "output_mode":
            values = line_list[1:]
            mode = ["full", "full_nobb", "simplified"]
            for value in values:
                if value in mode:
                    output_mode[value] = True
                else:
                    print("output_mode '%s' is not recognized" % (value))
                    sys.exit(1)

        elif option == "simplified_outfile":
            simplified_outfile = line_list[1]
        elif option == "full_outfile":
            full_outfile = line_list[1]
        elif option == "full_nobb_outfile":
            full_nobb_outfile = line_list[1]
        elif option == "sim_outfile":
            sim_outfile = line_list[1]
        elif option == "logfile":
            logfile = line_list[1]
        elif option:
            print("Warning: '%s' option is not recognized" % (option))

    # Check output_mode value, if undefined then assign default value
    if output_mode == output_mode_undefined:
        output_mode['full'] = True

    parse_result = {
        'docking_method': docking_method,
        'docking_conf': docking_conf,
        'similarity_coef': similarity_coef,
        'full_ref': full_ref,
        'full_nobb_ref': full_nobb_ref,
        'simplified_ref': simplified_ref,
        'use_backbone': use_backbone,
        'residue_name': res_name,
        'residue_number': res_num,
        'res_weight1': res_weight1,
        'res_weight2': res_weight2,
        'res_weight3': res_weight3,
        'res_weight4': res_weight4,
        'res_weight5': res_weight5,
        'docking_score': docking_score,
        'output_mode': output_mode,
        'simplified_outfile': simplified_outfile,
        'full_outfile': full_outfile,
        'full_nobb_outfile': full_nobb_outfile,
        'sim_outfile': sim_outfile,
        'logfile': logfile,
    }
    '''import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(parse_result)'''
    return parse_result
