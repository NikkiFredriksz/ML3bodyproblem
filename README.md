# ML3bodyproblem notulen
19-11: Olaf: data geplot, gekijkt waar data t meeste verschilt. 
	-a_pb scheidt t beste, maar maar voor 1 procent dat t echt wordt gescheiden
-verdeling in features geanalyseerd, komt ws gewoon overeen met wat in paper staat met hoe data is gegenereerd (natuurkundige argumenten voor density distributions)
-klasse 3 verschilt t meeste dus die is het makkelijkst te scheiden

26-11: Olaf: XGBoost resultaten geupload; 75 hyperparameter optimisation trials, classificatie 0 vs 1 vs 2 vs 3. 75 hyperparameter optimisation trials, binary classificatie 012 vs 3 (met extra weights op 3). 75 hyperparameter optimisation trials, binary classificatie 023 vs 1. Wat nog gedaan moet worden: 75 hyperparameter optimisation trials, classificatie 0 vs 1 vs 2 vs 3 met weights. 75 hyperparameter optimisation trials, classificatie 0 vs 1 vs 2, met en zonder weights.

makkelijkste manier om data in te lezen (werkt voor GC, YSC, dat soort zonder de extra kolom)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# We load the data
file = r"C:\Users\Lenovo T14 Gen 2\ML3body\ML3bodyproblem\GCdata\testGC.dat" #hier je eigen adres waar je bestanden staan
df = pd.read_csv(file, delim_whitespace=True, header=1)

# Geef kolomnamen
df.columns = [
    "N_sim", "m1", "m2", "m3", "a_pc", "e", "b_max_pc",
    "phi", "theta", "psi", "f", "v_km_s", "Ecc_Anomaly",
    "a_hard_pc", "a_ej_pc", "a_gw_pc", "t_coal_yr",
    "OUTCOME", "MERGE", "2gBBH", "TRIPLE"
]
