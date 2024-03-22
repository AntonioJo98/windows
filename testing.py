import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

folder_path = '/home/antonio98/Desktop/Projeto_em_Bioquímica/Training_set'  # Replace with the path to your folder containing SDF files

# Define a function to extract the molecule number from the filename
def extract_molecule_number(file_name):
    return file_name.split('_')[-1].split('.')[0]  # Assumes the number is the last part of the filename before the extension

# Create an empty dictionary to store descriptors for each molecule
all_descriptors = {}

# Iterate over all SDF files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.sdf'):
        file_path = os.path.join(folder_path, file_name)
        suppl = Chem.SDMolSupplier(file_path, removeHs=False)

        # Extract molecule number from filename
        molecule_number = extract_molecule_number(file_name)

        for mol in suppl:
            if mol is None:
                break
            molH = Chem.AddHs(mol)
            Chem.SanitizeMol(molH)
            descriptors = {
                'Chi0': Descriptors.Chi0(molH),
                'Chi1': Descriptors.Chi1(molH),
                des2 = Descriptors.Kappa1(molH)
                des3 = Descriptors.Kappa2(molH)
                des4 = Descriptors.Kappa3(molH)
                des5 = Descriptors.HallKierAlpha(molH)
                des6 = Descriptors.Ipc(molH)
                des7 = Descriptors.BertzCT(molH)
                des8 = Descriptors.BalabanJ(molH)
                des9 = Descriptors.MolLogP(molH)
                des10 = Descriptors.TPSA(molH)
                des13 = ComputeGasteigerCharges(molH)
                des14 = Descriptors.Chi0n(molH)
                des15 = Descriptors.Chi1n(molH)
                des16 = Descriptors.Chi2n(molH)
                des17 = Descriptors.Chi3n(molH)
                des18 = Descriptors.Chi4n(molH)
                des19 = Descriptors.Chi0v(molH)
                des20 = Descriptors.Chi1v(molH)
                des21 = Descriptors.Chi2v(molH)
                des22 = Descriptors.Chi3v(molH)
                des23 = Descriptors.Chi4v(molH)
                des24 = Descriptors.HeavyAtomCount(molH)
                des25 = Descriptors.HeavyAtomMolWt(molH)
                des26 = Descriptors.NHOHCount(molH)
                des27 = Descriptors.NOCount(molH)
                des28 = Descriptors.NumHAcceptors(molH)
                des29 = Descriptors.NumHDonors(molH)
                des30 = Descriptors.NumHeteroatoms(molH)
                des31 = Descriptors.NumRotatableBonds(molH)
                des32 = Descriptors.NumValenceElectrons(molH)
                des33 = rdMolDescriptors.CalcNumAmideBonds(molH)
                des34 = rdMolDescriptors.CalcNumAromaticRings(molH)
                des35 = rdMolDescriptors.CalcNumSaturatedRings(molH)
                des36 = rdMolDescriptors.CalcNumAliphaticRings(molH)
                des37 = rdMolDescriptors.CalcNumHeterocycles(molH)
                des38 =  Descriptors.RingCount(molH)
                des39 = Descriptors.FractionCSP3(molH)
                des40 = rdMolDescriptors.CalcNumSpiroAtoms(molH)
                des41 = rdMolDescriptors.CalcNumBridgeheadAtoms(molH)
                des42 = Descriptors.LabuteASA(molH)
                des43 = Descriptors.PEOE_VSA1(molH)
                des44 = Descriptors.PEOE_VSA2(molH)
                des45 = Descriptors.PEOE_VSA3(molH)
                des46 = Descriptors.PEOE_VSA4(molH)
                des47 = Descriptors.PEOE_VSA5(molH)
                des48 = Descriptors.PEOE_VSA6(molH)
                des49 = Descriptors.PEOE_VSA7(molH)
                des50 = Descriptors.PEOE_VSA8(molH)
                des51 = Descriptors.PEOE_VSA9(molH)
                des52 = Descriptors.PEOE_VSA10(molH)
                des53 = Descriptors.PEOE_VSA11(molH)
                des54 = Descriptors.PEOE_VSA12(molH)
                des55 = Descriptors.PEOE_VSA13(molH)
                des56 = Descriptors.PEOE_VSA14(molH)
                des57 = Descriptors.SMR_VSA1(molH)
                des58 = Descriptors.SMR_VSA2(molH)
                des59 = Descriptors.SMR_VSA3(molH)
                des60 = Descriptors.SMR_VSA4(molH)
                des61 = Descriptors.SMR_VSA5(molH)
                des62 = Descriptors.SMR_VSA6(molH)
                des63 = Descriptors.SMR_VSA7(molH)
                des64 = Descriptors.SMR_VSA8(molH)
                des65 = Descriptors.SMR_VSA9(molH)
                des66 = Descriptors.SMR_VSA10(molH)
                des67 = Descriptors.SlogP_VSA1(molH)
                des68 = Descriptors.SlogP_VSA2(molH)
                des69 = Descriptors.SlogP_VSA3(molH)
                des70 = Descriptors.SlogP_VSA4(molH)
                des71 = Descriptors.SlogP_VSA5(molH)
                des72 = Descriptors.SlogP_VSA6(molH)
                des73 = Descriptors.SlogP_VSA7(molH)
                des74 = Descriptors.SlogP_VSA8(molH)
                des75 = Descriptors.SlogP_VSA9(molH)
                des76 = Descriptors.SlogP_VSA10(molH)
                des77 = Descriptors.SlogP_VSA11(molH)
                des78 = Descriptors.SlogP_VSA12(molH)
                des79 = Descriptors.EState_VSA1(molH)
                des80 = Descriptors.EState_VSA2(molH)
                des81 = Descriptors.EState_VSA3(molH)
                des82 = Descriptors.EState_VSA4(molH)
                des83 = Descriptors.EState_VSA5(molH)
                des84 = Descriptors.EState_VSA6(molH)
                des85 = Descriptors.EState_VSA7(molH)
                des86 = Descriptors.EState_VSA8(molH)
                des87 = Descriptors.EState_VSA9(molH)
                des88 = Descriptors.EState_VSA10(molH)
                des89 = Descriptors.EState_VSA11(molH)
                des90 = Descriptors.VSA_EState1(molH)
                des91 = Descriptors.VSA_EState2(molH)
                des92 = Descriptors.VSA_EState3(molH)
                des93 = Descriptors.VSA_EState4(molH)
                des94 = Descriptors.VSA_EState5(molH)
                des95 = Descriptors.VSA_EState6(molH)
                des96 = Descriptors.VSA_EState7(molH)
                des97 = Descriptors.VSA_EState8(molH)
                des98 = Descriptors.VSA_EState9(molH)
                des99 = Descriptors.VSA_EState10(molH)
                des101 = Fragments.fr_Al_COO(molH)
                des102 = Fragments.fr_Al_OH(molH)
                des103 = Fragments.fr_Al_OH_noTert(molH)
                des104 = Fragments.fr_ArN(molH)
                des105 = Fragments.fr_Ar_COO(molH)
                des106 = Fragments.fr_Ar_N(molH)
                des107 = Fragments.fr_Ar_NH(molH)
                des108 = Fragments.fr_Ar_OH(molH)
                des109 = Fragments.fr_COO(molH)
                des110 = Fragments.fr_COO2(molH)
                des111 = Fragments.fr_C_O(molH)
                des112 = Fragments.fr_C_O_noCOO(molH)
                des113 = Fragments.fr_C_S(molH)
                des114 = Fragments.fr_HOCCN(molH)
                des115 = Fragments.fr_Imine(molH)
                des116 = Fragments.fr_NH0(molH)
                des117 = Fragments.fr_NH1(molH)
                des118 = Fragments.fr_NH2(molH)
                des119 = Fragments.fr_N_O(molH)
                des120 = Fragments.fr_Ndealkylation1(molH)
                des121 = Fragments.fr_Ndealkylation2(molH)
                des122 = Fragments.fr_Nhpyrrole(molH)
                des123 = Fragments.fr_SH(molH)
                des124 = Fragments.fr_aldehyde(molH)
                des125 = Fragments.fr_alkyl_carbamate(molH)
                des126 = Fragments.fr_alkyl_halide(molH)
                des127 = Fragments.fr_allylic_oxid(molH)
                des128 = Fragments.fr_amide(molH)
                des129 = Fragments.fr_amidine(molH)
                des130 = Fragments.fr_aniline(molH)
                des131= Fragments.fr_aryl_methyl(molH)
                des132 = Fragments.fr_azide(molH)
                des133 = Fragments.fr_azo(molH)
                des134 = Fragments.fr_barbitur(molH)
                des135 = Fragments.fr_benzene(molH)
                des136 = Fragments.fr_benzodiazepine(molH)
                des137 = Fragments.fr_bicyclic(molH)
                des138 = Fragments.fr_diazo(molH)
                des139 = Fragments.fr_dihydropyridine(molH)
                des140 = Fragments.fr_epoxide(molH)
                des141 = Fragments.fr_ester(molH)
                des142 = Fragments.fr_ether(molH)
                des143= Fragments.fr_furan(molH)
                des144 = Fragments.fr_guanido(molH)
                des145 = Fragments.fr_halogen(molH)
                des146 = Fragments.fr_hdrzine(molH)
                des147 = Fragments.fr_hdrzone(molH)
                des148 = Fragments.fr_imidazole(molH)
                des149 = Fragments.fr_imide(molH)
                des150 = Fragments.fr_isocyan(molH)
                des151 = Fragments.fr_isothiocyan(molH)
                des152 = Fragments.fr_ketone(molH)
                des153 = Fragments.fr_ketone_Topliss(molH)
                des154 = Fragments.fr_lactam(molH)
                des155 = Fragments.fr_lactone(molH)
                des156 = Fragments.fr_methoxy(molH)
                des157 = Fragments.fr_morpholine(molH)
                des158 = Fragments.fr_nitrile(molH)
                des159 = Fragments.fr_nitro(molH)
                des160= Fragments.fr_nitro_arom(molH)
                des161 = Fragments.fr_nitro_arom_nonortho(molH)
                des162 = Fragments.fr_nitroso(molH)
                des163 = Fragments.fr_oxazole(molH)
                des164 = Fragments.fr_oxime(molH)
                des165 = Fragments.fr_para_hydroxylation(molH)
                des166 = Fragments.fr_phenol(molH)
                des167 = Fragments.fr_phenol_noOrthoHbond(molH)
                des168 = Fragments.fr_phos_acid(molH)
                des169 = Fragments.fr_phos_ester(molH)
                des170 = Fragments.fr_piperdine(molH)
                des171 = Fragments.fr_piperzine(molH)
                des172 = Fragments.fr_priamide(molH)
                des173 = Fragments.fr_prisulfonamd(molH)
                des174 = Fragments.fr_pyridine(molH)
                des175 = Fragments.fr_quatN(molH)
                des176 = Fragments.fr_sulfide(molH)
                des177 = Fragments.fr_sulfonamd(molH)
                des178 = Fragments.fr_sulfone(molH)
                des179 = Fragments.fr_term_acetylene(molH)
                des180 = Fragments.fr_tetrazole(molH)
                des181 = Fragments.fr_thiazole(molH)
                des182 = Fragments.fr_thiocyan(molH)
                des183 = Fragments.fr_thiophene(molH)
                des184 = Fragments.fr_unbrch_alkane(molH)
                des185 = Fragments.fr_urea(molH)
            }
            # Add descriptors to the dictionary
            all_descriptors[molecule_number] = descriptors

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(all_descriptors, orient='index')

# Save DataFrame to CSV file
output_csv = '/home/antonio98/Desktop/Projeto_em_Bioquímica/descriptors.csv'  # Replace with your desired output file path
df.to_csv(output_csv)