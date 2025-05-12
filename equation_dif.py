import matplotlib.pyplot as plt
import numpy as np 

time = [0]
lapin = [1]
renard = [2]

alpha = 1/3
beta = 1/3
gamma = 1/3
delta = 1/3

step = 0.001


for _ in range(1, 100_000):
    time_update = time[-1] + step
    lapin_update = (lapin[-1]*(alpha - beta * renard[-1])) * step + lapin[-1]
    renard_update = (renard[-1]*(delta*lapin[-1] - gamma)) * step  + renard[-1]

    lapin.append(lapin_update)
    renard.append(renard_update)
    time.append(time_update)

lapin = np.array(lapin)
lapin *= 1000
renard = np.array(renard)
renard *= 1000 

plt.figure(figsize=(15, 6))
plt.plot(time, lapin, "b-", label = 'Lapin')
plt.plot(time, renard, "r-", label = 'Rendars')
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title('Proie_Predateur')
plt.show()

