import matplotlib.pyplot as plt
import numpy as np

x = np.arange(11)

y1=[0.1918, 0.2607, 0.2989, 0.3263, 0.3417, 0.3537, 0.3628, 0.3718, 0.3809, 0.3867, 0.3897]
y2=[0.1128, 0.0681, 0.1121, 0.1348, 0.154, 0.1699, 0.1451, 0.1573, 0.1685, 0.1716, 0.1739]
y3=[0.0299, 0.1186, 0.0992, 0.0889, 0.0744, 0.0633, 0.0922, 0.0831, 0.0746, 0.0723, 0.0699]
y4=[0.02, 0.0248, 0.0275, 0.0295, 0.0326, 0.0345, 0.0397, 0.0407, 0.0424, 0.0431, 0.0428]
y5=[0.1447, 0.189, 0.2143, 0.2269, 0.2321, 0.2367, 0.2354, 0.2381, 0.2391, 0.2392, 0.2395]
y6=[0.0857, 0.0304, 0.0685, 0.0885, 0.1085, 0.1237, 0.0974, 0.1096, 0.121, 0.1244, 0.1271]


#x=np.array([0,1,2,3,4,5,6,7,8,9,10,11])
my_ticks=['800','1200', '1600', '2000', '2400', '2800', '3200', '3600', '4000', '4400', '4600']
plt.xticks(x, my_ticks)
plt.xlabel("Iterations (thousands)")
plt.ylabel("regret")
plt.title("Epileptic Seizure Shuffle Absolute Regret Comparison per Method")

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.plot(x, y5)
plt.plot(x, y6)

plt.legend(['LR', 'Perceptron', 'RBF-Perceptron', 'RBF-SVM', 'SVM', 'Winnow'], loc='upper left')

plt.show()