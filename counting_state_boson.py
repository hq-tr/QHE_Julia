import sys
sys.path.append("/home/trung/_qhe-library/")
import task1
from scipy.linalg import eig
from angular_momentum import LminusLplus_boson
import numpy as np
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("--N_el", "-e", type=int, default=4, help="Number of bosons")
ap.add_argument("--N_orb", "-o", type=int, default=5, help="Number of orbitals")

aa = ap.parse_args()
Ne = aa.N_el
No = aa.N_orb

S = (No-1)/2.
Lz_max = int(Ne*S)
def countstates(Lz):
	b = task1.findBasis_brute(Ne,No,True,Lz=Lz, bosonic=True)
	#L = LminusLplus_boson(b,No)
	#E, V = eig(L.toarray())
	#print(E)
	#print(b)
	return len(b)

cs = np.array(list(map(countstates, range(Ne*(No-1)//2+5))))

cshw = cs[:-1]-cs[1:]
for i in range(Lz_max+1):
	#c = counthighestweight(4,10,i)
	print(f"{i}\t{cshw[i]}")