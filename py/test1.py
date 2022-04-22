import matplotlib.pyplot as plt

d = [x for x in range(10)]
plt.plot(d,[k*2+1 for k in d])
plt.show()