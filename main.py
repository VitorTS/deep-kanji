import data_loader
import numpy as np

def dd(val):
    print(val)
    quit()
    
def sig(z):
    return 1 / (1 + np.exp(-z))
    
#training_data , validation_data , test_data = load_data_wrapper()
training_data = data_loader.load_data_wrapper()

x = training_data[0] 

layerSizes = [len(x[0]), 15, 10];

#feed the numbers in layerSizes as sizes for randn
biases = [np.random.normal(0.0, 1.0, (row, 1)) for row in layerSizes[1:]]
#dd(biases)

fullLayer = [(len(x[0]), 15), (15, 10)]
weights = [np.random.normal(0.0, 1.0/np.sqrt(col), (row, col)) for col, row in fullLayer]
#dd(weights[0][0]) #peek to see if the weights were set correctly

a = x[0]
#dd(a)

b = biases[0]
w = weights[0]

#dd( (a * w).shape )
#dd( [w.shape, a.shape ] )

#z = np.dot(w, a) - b
#dd( [z,sig(z)] )


for wk in w:
    dd(wk)
    
for w, b in zip(biases, weights):
    z = w * a - b
    a = sig(z)
	