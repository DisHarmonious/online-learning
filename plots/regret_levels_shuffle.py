import matplotlib.pyplot as plt
import numpy as np
 
value1=[0.2994, 0.4031, 0.4589, 0.4921, 0.5169, 0.534, 0.5479, 0.5591, 0.5687, 0.5772, 0.5807]
value2=[0.2452, 0.3265, 0.3686, 0.3924, 0.4075, 0.4187, 0.4277, 0.4342, 0.4389, 0.4429, 0.4442]
value3=[0.1027, 0.1398, 0.1573, 0.1687, 0.1791, 0.1856, 0.1905, 0.1938, 0.1958, 0.199, 0.2004]
value4=[0.0765, 0.1092, 0.1265, 0.1359, 0.1459, 0.1521, 0.1559, 0.1591, 0.1607, 0.1644, 0.1658]
value5=[0.2412, 0.323, 0.3684, 0.3923, 0.4106, 0.4233, 0.431, 0.4379, 0.4422, 0.4466, 0.4481]
value6=[0.2182, 0.2887, 0.325, 0.3461, 0.362, 0.3725, 0.3801, 0.3864, 0.3914, 0.3958, 0.3974]


box_plot_data=[value1,value2,value3,value4, value5, value6]
plt.boxplot(box_plot_data)
plt.title('Epileptic Seizure regret after shuffling')
x=np.arange(7)
plt.xticks(x, (' ','LR', 'Perceptron', 'RBF-Perceptron', 'RBF-SVM', 'SVM', 'Winnow'))
plt.xlabel('Method')
plt.ylabel('regret')
plt.show()