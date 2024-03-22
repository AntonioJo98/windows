#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

# Define a function to calculate all descriptors for a given molecule
def calculate_descriptors(mol):
    molH = Chem.AddHs(mol)
    Chem.SanitizeMol(molH)
    
    descriptors = {
        'Chi0': Descriptors.Chi0(molH),
        'Chi1': Descriptors.Chi1(molH),
        'Kappa1': Descriptors.Kappa1(molH),
        'Kappa2': Descriptors.Kappa2(molH),
        'Kappa3': Descriptors.Kappa3(molH),
        'HallKierAlpha': Descriptors.HallKierAlpha(molH),
        'Ipc': Descriptors.Ipc(molH),
        'BertzCT': Descriptors.BertzCT(molH),
        'BalabanJ': Descriptors.BalabanJ(molH),
        'MolLogP': Descriptors.MolLogP(molH),
        'TPSA': Descriptors.TPSA(molH),
        'ComputeGasteigerCharges': ComputeGasteigerCharges(molH),
        'Chi0n': Descriptors.Chi0n(molH),
        'Chi1n': Descriptors.Chi1n(molH),
        'Chi2n': Descriptors.Chi2n(molH),
        'Chi3n': Descriptors.Chi3n(molH),
        'Chi4n': Descriptors.Chi4n(molH),
        'Chi0v': Descriptors.Chi0v(molH),
        'Chi1v': Descriptors.Chi1v(molH),
        'Chi2v': Descriptors.Chi2v(molH),
        'Chi3v': Descriptors.Chi3v(molH),
        'Chi4v': Descriptors.Chi4v(molH),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(molH),
        'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(molH),
        'NHOHCount': Descriptors.NHOHCount(molH),
        'NOCount': Descriptors.NOCount(molH),
        'NumHAcceptors': Descriptors.NumHAcceptors(molH),
        'NumHDonors': Descriptors.NumHDonors(molH),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(molH),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molH),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(molH),
        'CalcNumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molH),
        'CalcNumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molH),
        'CalcNumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molH),
        'CalcNumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molH),
        'CalcNumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molH),
        'RingCount': Descriptors.RingCount(molH),
        'FractionCSP3': Descriptors.FractionCSP3(molH),
        'CalcNumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molH),
        'CalcNumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molH),
        'LabuteASA': Descriptors.LabuteASA(molH),
        'PEOE_VSA1': Descriptors.PEOE_VSA1(molH),
        'PEOE_VSA2': Descriptors.PEOE_VSA2(molH),
        'PEOE_VSA3': Descriptors.PEOE_VSA3(molH),
        'PEOE_VSA4': Descriptors.PEOE_VSA4(molH),
        'PEOE_VSA5': Descriptors.PEOE_VSA5(molH),
        'PEOE_VSA6': Descriptors.PEOE_VSA6(molH),
        'PEOE_VSA7': Descriptors.PEOE_VSA7(molH),
        'PEOE_VSA8': Descriptors.PEOE_VSA8(molH),
        'PEOE_VSA9': Descriptors.PEOE_VSA9(molH),
        'PEOE_VSA10': Descriptors.PEOE_VSA10(molH),
        'PEOE_VSA11': Descriptors.PEOE_VSA11(molH),
        'PEOE_VSA12': Descriptors.PEOE_VSA12(molH),
        'PEOE_VSA13': Descriptors.PEOE_VSA13(molH),
        'PEOE_VSA14': Descriptors.PEOE_VSA14(molH),
        'SMR_VSA1': Descriptors.SMR_VSA1(molH),
        'SMR_VSA2': Descriptors.SMR_VSA2(molH),
        'SMR_VSA3': Descriptors.SMR_VSA3(molH),
        'SMR_VSA4': Descriptors.SMR_VSA4(molH),
        'SMR_VSA5': Descriptors.SMR_VSA5(molH),
        'SMR_VSA6': Descriptors.SMR_VSA6(molH),
        'SMR_VSA7': Descriptors.SMR_VSA7(molH),
        'SMR_VSA8': Descriptors.SMR_VSA8(molH),
        'SMR_VSA9': Descriptors.SMR_VSA9(molH),
        'SMR_VSA10': Descriptors.SMR_VSA10(molH),
        'SlogP_VSA1': Descriptors.SlogP_VSA1(molH),
        'SlogP_VSA2': Descriptors.SlogP_VSA2(molH),
        'SlogP_VSA3': Descriptors.SlogP_VSA3(molH),
        'SlogP_VSA4': Descriptors.SlogP_VSA4(molH),
        'SlogP_VSA5': Descriptors.SlogP_VSA5(molH),
        'SlogP_VSA6': Descriptors.SlogP_VSA6(molH),
        'SlogP_VSA7': Descriptors.SlogP_VSA7(molH),
        'SlogP_VSA8': Descriptors.SlogP_VSA8(molH),
        'SlogP_VSA9': Descriptors.SlogP_VSA9(molH),
        'SlogP_VSA10': Descriptors.SlogP_VSA10(molH),
        'SlogP_VSA11': Descriptors.SlogP_VSA11(molH),
        'SlogP_VSA12': Descriptors.SlogP_VSA12(molH),
        'EState_VSA1': Descriptors.EState_VSA1(molH),
        'EState_VSA2': Descriptors.EState_VSA2(molH),
        'EState_VSA3': Descriptors.EState_VSA3(molH),
        'EState_VSA4': Descriptors.EState_VSA4(molH),
        'EState_VSA5': Descriptors.EState_VSA5(molH),
        'EState_VSA6': Descriptors.EState_VSA6(molH),
        'EState_VSA7': Descriptors.EState_VSA7(molH),
        'EState_VSA8': Descriptors.EState_VSA8(molH),
        'EState_VSA9': Descriptors.EState_VSA9(molH),
        'EState_VSA10': Descriptors.EState_VSA10(molH),
        'EState_VSA11': Descriptors.EState_VSA11(molH),
        'VSA_EState1': Descriptors.VSA_EState1(molH),
        'VSA_EState2': Descriptors.VSA_EState2(molH),
        'VSA_EState3': Descriptors.VSA_EState3(molH),
        'VSA_EState4': Descriptors.VSA_EState4(molH),
        'VSA_EState5': Descriptors.VSA_EState5(molH),
        'VSA_EState6': Descriptors.VSA_EState6(molH),
        'VSA_EState7': Descriptors.VSA_EState7(molH),
        'VSA_EState8': Descriptors.VSA_EState8(molH),
        'VSA_EState9': Descriptors.VSA_EState9(molH),
        'VSA_EState10': Descriptors.VSA_EState10(molH),
        'fr_Al_COO': Fragments.fr_Al_COO(molH),
        'fr_Al_OH': Fragments.fr_Al_OH(molH),
        'fr_Al_OH_noTert': Fragments.fr_Al_OH_noTert(molH),
        'fr_ArN': Fragments.fr_ArN(molH),
        'fr_Ar_COO': Fragments.fr_Ar_COO(molH),
        'fr_Ar_N': Fragments.fr_Ar_N(molH),
        'fr_Ar_NH': Fragments.fr_Ar_NH(molH),
        'fr_Ar_OH': Fragments.fr_Ar_OH(molH),
        'fr_COO': Fragments.fr_COO(molH),
        'fr_COO2': Fragments.fr_COO2(molH),
        'fr_C_O': Fragments.fr_C_O(molH),
        'fr_C_O_noCOO': Fragments.fr_C_O_noCOO(molH),
        'fr_C_S': Fragments.fr_C_S(molH),
        'fr_HOCCN': Fragments.fr_HOCCN(molH),
        'fr_Imine': Fragments.fr_Imine(molH),
        'fr_NH0': Fragments.fr_NH0(molH),
        'fr_NH1': Fragments.fr_NH1(molH),
        'fr_NH2': Fragments.fr_NH2(molH),
        'fr_N_O': Fragments.fr_N_O(molH),
        'fr_Ndealkylation1': Fragments.fr_Ndealkylation1(molH),
        'fr_Ndealkylation2': Fragments.fr_Ndealkylation2(molH),
        'fr_Nhpyrrole': Fragments.fr_Nhpyrrole(molH),
        'fr_SH': Fragments.fr_SH(molH),
        'fr_aldehyde': Fragments.fr_aldehyde(molH),
        'fr_alkyl_carbamate': Fragments.fr_alkyl_carbamate(molH),
        'fr_alkyl_halide': Fragments.fr_alkyl_halide(molH),
        'fr_allylic_oxid': Fragments.fr_allylic_oxid(molH),
        'fr_amide': Fragments.fr_amide(molH),
        'fr_amidine': Fragments.fr_amidine(molH),
        'fr_aniline': Fragments.fr_aniline(molH),
        'fr_aryl_methyl': Fragments.fr_aryl_methyl(molH),
        'fr_azide': Fragments.fr_azide(molH),
        'fr_azo': Fragments.fr_azo(molH),
        'fr_barbitur': Fragments.fr_barbitur(molH),
        'fr_benzene': Fragments.fr_benzene(molH),
        'fr_benzodiazepine': Fragments.fr_benzodiazepine(molH),
        'fr_bicyclic': Fragments.fr_bicyclic(molH),
        'fr_diazo': Fragments.fr_diazo(molH),
        'fr_dihydropyridine': Fragments.fr_dihydropyridine(molH),
        'fr_epoxide': Fragments.fr_epoxide(molH),
        'fr_ester': Fragments.fr_ester(molH),
        'fr_ether': Fragments.fr_ether(molH),
        'fr_furan': Fragments.fr_furan(molH),
        'fr_guanido': Fragments.fr_guanido(molH),
        'fr_halogen': Fragments.fr_halogen(molH),
        'fr_hdrzine': Fragments.fr_hdrzine(molH),
        'fr_hdrzone': Fragments.fr_hdrzone(molH),
        'fr_imidazole': Fragments.fr_imidazole(molH),
        'fr_imide': Fragments.fr_imide(molH),
        'fr_isocyan': Fragments.fr_isocyan(molH),
        'fr_isothiocyan': Fragments.fr_isothiocyan(molH),
        'fr_ketone': Fragments.fr_ketone(molH),
        'fr_ketone_Topliss': Fragments.fr_ketone_Topliss(molH),
        'fr_lactam': Fragments.fr_lactam(molH),
        'fr_lactone': Fragments.fr_lactone(molH),
        'fr_methoxy': Fragments.fr_methoxy(molH),
        'fr_morpholine': Fragments.fr_morpholine(molH),
        'fr_nitrile': Fragments.fr_nitrile(molH),
        'fr_nitro': Fragments.fr_nitro(molH),
        'fr_nitro_arom': Fragments.fr_nitro_arom(molH),
        'fr_nitro_arom_nonortho': Fragments.fr_nitro_arom_nonortho(molH),
        'fr_nitroso': Fragments.fr_nitroso(molH),
        'fr_oxazole': Fragments.fr_oxazole(molH),
        'fr_oxime': Fragments.fr_oxime(molH),
        'fr_para_hydroxylation': Fragments.fr_para_hydroxylation(molH),
        'fr_phenol': Fragments.fr_phenol(molH),
        'fr_phenol_noOrthoHbond': Fragments.fr_phenol_noOrthoHbond(molH),
        'fr_phos_acid': Fragments.fr_phos_acid(molH),
        'fr_phos_ester': Fragments.fr_phos_ester(molH),
        'fr_piperdine': Fragments.fr_piperdine(molH),
        'fr_piperzine': Fragments.fr_piperzine(molH),
        'fr_priamide': Fragments.fr_priamide(molH),
        'fr_prisulfonamd': Fragments.fr_prisulfonamd(molH),
        'fr_pyridine': Fragments.fr_pyridine(molH),
        'fr_quatN': Fragments.fr_quatN(molH),
        'fr_sulfide': Fragments.fr_sulfide(molH),
        'fr_sulfonamd': Fragments.fr_sulfonamd(molH),
        'fr_sulfone': Fragments.fr_sulfone(molH),
        'fr_term_acetylene': Fragments.fr_term_acetylene(molH),
        'fr_tetrazole': Fragments.fr_tetrazole(molH),
        'fr_thiazole': Fragments.fr_thiazole(molH),
        'fr_thiocyan': Fragments.fr_thiocyan(molH),
        'fr_thiophene': Fragments.fr_thiophene(molH),
        'fr_unbrch_alkane': Fragments.fr_unbrch_alkane(molH),
        'fr_urea': Fragments.fr_urea(molH)
        }
    return descriptors
# Function to process all SDF files in a folder
def process_sdf_folder(folder_path):
    all_descriptors = []  # Initialize a list to store descriptors for all molecules
    
    # Process each SDF file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.sdf'):
            file_path = os.path.join(folder_path, file_name)
            suppl = Chem.SDMolSupplier(file_path, removeHs=False)
            
            # Process each molecule in the SDF file
            for idx, mol in enumerate(suppl):
                if mol is None:
                    break
                
                # Calculate descriptors for the molecule
                descriptors = calculate_descriptors(mol)
                
                # Extract the molecule number from the file name
                molecule_number = os.path.splitext(file_name)[0]
                
                # Append the descriptors to the list
                all_descriptors.append([molecule_number] + list(descriptors.values()))
    
    return all_descriptors

# Set the folder containing SDF files
sdf_folder = '/home/antonio98/Desktop/Projeto_em_Bioquímica/Training_set'  # Adjust the folder path as needed

# Process all SDF files in the folder
all_descriptors = process_sdf_folder(sdf_folder)

# Create a DataFrame from the list of descriptors
columns = ['Molecule_Number'] + list(all_descriptors[0][1:])
df = pd.DataFrame(all_descriptors, columns=columns)

# Save the DataFrame to a CSV file
output_file = '/home/antonio98/Desktop/Projeto_em_Bioquímica/descriptors2_output.csv'  # Adjust the file name as needed
df.to_csv(output_file, index=False)

print(f"Descriptors calculated for {len(df)} molecules. Output saved to '{output_file}'.")
