import numpy as np
import sparse_ir as ir
import t2g_soc_jtpol as tsjt
import os
from datetime import datetime
import csv




data_dir = "t2g_soc_jtpol_data"

if data_dir not in os.listdir():
    os.mkdir(data_dir)


Nls = [1,2,3,4,5]
tls = [0.05,0.2,1.2]
Usocls = [(4,0.8,0.05), (2.5,0.2,0.3), (0.5,0.04,0.3), (4,0.8,0), (2.5,0.2,0), (0,0,0.3)]
gls = [(0.1,0.1), (0,0), (0,0.1)]
Tls = [10,4]
vals = [(T,8,N,t,U[0],U[1],g[0],0.1,g[1],U[2],24,5) for T in Tls for g in gls for U in Usocls for t in tls for N in Nls]

with open("simulated_values.csv", mode='r+', newline='') as csvfl:
    reader = csv.reader(csvfl)
    lines = [line[:-1] for l,line in enumerate(reader) if l!=0]
    writer = csv.writer(csvfl)
    # writer.writerow(['T', 'wm', 'N', 't', 'U', 'J', 'Jphm', 'w0', 'g', 'lbd', 'k_sz', 'diis_mem', 'now'])
    
    for val in vals:
        already_comp = False
        for line in lines:
            if val == tuple([float(s) for s in line]):
                already_comp = True
                break
        
        if not already_comp:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            dysolver = tsjt.DysonSolver(*val, fl=data_dir+'/'+now+".out")
            dysolver.solve(diis_active=True, tol=5e-6)
            dysolver.save(data_dir+'/'+now)
            writer.writerow(tuple(list(val)+[int(now)]))