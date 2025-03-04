"""
This python script combines the protein complex data from Hint and CORUM to identify the protein interaction structure of specific protein co-complexes.
The original datasets can be found at https://hint.yulab.org/ and https://mips.helmholtz-muenchen.de/corum/ respectively.
The first dataset contains protein-protein interactions for complexes. The complexes are not identified explicitly but are rather matched to a PubMedID.
The second dataset contains the names and id's of protein complexes as well as their PubMedID.
We merge the datasets based on PubMedID and then use the Uniprot ID's to distinguish the interactions corresponding with which complex_id in the article.
The resulting dataset has the information from each individual dataset where each row is a protein interaction and the complex_id denotes which complex contains the said protein interaction.
"""

import pandas as pd
import argparse

# Accept input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--corum_path',type=str,
                    default='/work/ccnr/glover.co/net_design/NetDesign/data/protein_complex/corum_allComplexes.txt',
                    help='File path for corum dataset')
parser.add_argument('--hint_path',type=str,
                    default='/work/ccnr/glover.co/net_design/NetDesign/data/protein_complex/both_all.txt',
                    help='File path for HINT dataset')
parser.add_argument('--path_to_write',type=str,
                    default='/work/ccnr/glover.co/net_design/NetDesign/data/protein_complex/corum_hint_combined.txt',
                    help='File to store dataframe')

args = parser.parse_args()
corum_path = args.corum_path
hint_path = args.hint_path
file = args.path_to_write

# Read in datasets
hint = pd.read_csv(hint_path,delimiter='\t')
corum = pd.read_csv(corum_path,delimiter='\t')

# Clean HINT dataset
hint['pmid:method:quality:type'] = hint['pmid:method:quality:type'].str.split('|')
hint = hint.explode('pmid:method:quality:type')
new_column_names = hint.columns[4].split(':')
hint[new_column_names] = hint[hint.columns[4]].str.split(':',expand=True)
hint.drop(columns=[hint.columns[4]],inplace=True)
hint = hint[hint['type'] == 'binary']
hint['pmid'] = hint['pmid'].astype(int)

# Merge datasets
combined = pd.merge(corum,hint,on='pmid')

# Assign correct complexes to protein interactions
combined['correct_complex'] = True
for row in range(len(combined)):
    cur_row = combined.iloc[row]
    if cur_row['Uniprot_A'] not in cur_row['subunits_uniprot_id'].split(';') or cur_row['Uniprot_B'] not in cur_row['subunits_uniprot_id'].split(';'):
        combined.iloc[row,-1] = False
combined = combined[combined['correct_complex'] == True]
combined = combined.drop_duplicates(subset=None)

combined.to_csv(file,index=False)