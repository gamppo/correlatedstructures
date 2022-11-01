import numpy as np
import pandas as pd
from correlationExtraction import CorrelationExtraction
from Bio.PDB.Polypeptide import is_aa
from sklearn.metrics import adjusted_mutual_info_score

#config
pdb_data = '/usr/users/olivia/data/PDB/DATA/data_PDB/1BUS.pdb'
Protein = pdb_data[41:-4]
model1 = CorrelationExtraction(
                 pdb_data,
                 mode='backbone',
                 nstates=2,
                 therm_fluct=0.5,
                 therm_iter=5,
                 loop_start=-1,
                 loop_end=-1)

#extract torsion angles
angle_data = model1.angCor.get_angle_data('A')

#ID conformer, ID# , Phi, Psi


#extract correlation values
resid1 = []

for res in model1.structure[0]['A'].get_residues():
    if is_aa(res, standard=True):
        resid1.append(res._id[1])

# aaS and aaF need to be defined in order to use the calc_ami function
model1.resid = resid1
model1.aaS = min(resid)
model1.aaF = max(resid)



ang_clusters, ang_banres = model1.angCor.clust_cor("A", resid1)
ang_ami, ang_hm = model1.calc_ami( ang_clusters, ang_banres)
cor_seq = np.mean(np.nan_to_num(ang_hm), axis=0)
model1.angCor = cor_seq



# final table: residue, phi, psi, ang correlation
#phi and psi still need to be calculated (average from the 5 therm_iter)
phi= np.zeros(len(model1.resid))
psi= np.zeros(len(model1.resid))


if model1.therm_iter >1:
    for i in range(model1.therm_iter):
        for j in range(len(model1.resid)):
            phi[j] += angle_data[i*len(model1.resid)+j,2]
            psi[j] += angle_data[i*len(model1.resid)+j,3]

phi_ang = phi/model1.therm_iter
psi_ang = psi/model1.therm_iter



#write a file with 4 columns

dict = {'Residue': resid1, 'Phi': phi_ang, 'Psi':psi_ang , 'AngCor': cor_seq}
df = pd.DataFrame(dict)
df.to_csv(f'/usr/users/olivia/PycharmProjects/correlatedstructures/InHouseStructures/{Protein}_angCorr')


#what is:
# banres
#
