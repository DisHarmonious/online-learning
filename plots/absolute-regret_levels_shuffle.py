import matplotlib.pyplot as plt
import numpy as np
 
value1=[0.1918, 0.2607, 0.2989, 0.3263, 0.3417, 0.3537, 0.3628, 0.3718, 0.3809, 0.3867, 0.3897]
value2=[0.1128, 0.0681, 0.1121, 0.1348, 0.154, 0.1699, 0.1451, 0.1573, 0.1685, 0.1716, 0.1739]
value3=[0.0299, 0.1186, 0.0992, 0.0889, 0.0744, 0.0633, 0.0922, 0.0831, 0.0746, 0.0723, 0.0699]
value4=[0.02, 0.0248, 0.0275, 0.0295, 0.0326, 0.0345, 0.0397, 0.0407, 0.0424, 0.0431, 0.0428]
value5=[0.1447, 0.189, 0.2143, 0.2269, 0.2321, 0.2367, 0.2354, 0.2381, 0.2391, 0.2392, 0.2395]
value6=[0.0857, 0.0304, 0.0685, 0.0885, 0.1085, 0.1237, 0.0974, 0.1096, 0.121, 0.1244, 0.1271]


box_plot_data=[value1,value2,value3,value4, value5, value6]
plt.boxplot(box_plot_data)
plt.title('Epileptic Seizure absolute regret after shuffling')
x=np.arange(7)
plt.xticks(x, (' ','LR', 'Perceptron', 'RBF-Perceptron', 'RBF-SVM', 'SVM', 'Winnow'))
plt.xlabel('Method')
plt.ylabel('regret')
plt.show()