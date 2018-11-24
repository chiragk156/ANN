import pickle
import matplotlib.pyplot as plt

infile = open('X_train','rb')
X_Graph = pickle.load(infile)
infile.close()

infile = open('Y_train','rb')
Y_Graph_train = pickle.load(infile)
infile.close()

infile = open('Y_test','rb')
Y_Graph_test = pickle.load(infile)
infile.close()

plt.plot(X_Graph, Y_Graph_train,'r',label = 'Training')
plt.plot(X_Graph, Y_Graph_test,'b',label = 'Validation')

# plt.ylim(0.2,0.5)
# naming the x axis 
plt.xlabel('No. of Epochs') 
# naming the y axis 
plt.ylabel('Mean Square Error') 
  
# giving a title to my graph 
# plt.title('Minibatch Size = 128 | Learning Rate = 0.01 | Dropout Rate = 0') 

plt.legend()
# function to show the plot 
plt.show()
