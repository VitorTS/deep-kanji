import data_loader
import numpy as np

def dd(val):
    print(val)
    quit()
    
def sig(z):
    return 1 / (1 + np.exp(-z))

def sig_prime(z):
    a = sig(z)
    return a * (1 - a)
    
def softmax(z):
    total = np.sum(np.exp(z))
    return [np.exp(zk) / total for zk in z]
    
#training_data , validation_data , test_data = load_data_wrapper()
training_data, test_data = data_loader.load_data_wrapper()

inputSize = len(training_data[0][0])
layerSizes = [inputSize, 15, 10];
learningRate = 0.1
batch_size = 30
epochs = 10

biases = [np.random.normal(0.0, 1.0, (row, 1)) for row in layerSizes[1:]]
#dd(biases)

fullLayer = [(inputSize, 15), (15, 10)]
weights = [np.random.normal(0.0, 1.0/np.sqrt(col), (row, col)) for col, row in fullLayer]

trainingBatches = []
for i in range(0, inputSize - batch_size, batch_size):
    trainingBatches.append( training_data[i: i + batch_size] )

for i in range(epochs):
    for batches in trainingBatches:
        gradient_w = [np.zeros((row, col)) for col, row in fullLayer];
        gradient_b = [np.zeros((row, 1)) for row in layerSizes[1:]];
        
        for a, y in batches:
            #b = biases[0]
            #w = weights[0]

            #dd( (a * w).shape )
            #dd( [w.shape, a.shape ] )

            #z = np.dot(w, a) - b
            #dd*( [z,sig(z)] )


            zetas = []
            activations = [a]
            for b, w in zip(biases, weights):
                z = np.dot(w, a) - b
                zetas.append(z)
                a = sig(z)
                activations.append(a)
                
            a = softmax(z)
            
            err = a - y
            #dd(err.shape)

            #dd( sig_prime(z) )
            #dd( weights[-1].shape )
            #dd( np.transpose(weights[-1]).shape )
            #dd( np.dot(np.transpose(weights[-1]), err).shape )
            #dd( sig_prime(zetas[-1]).shape )
             
            errors = [err]
            for w, z in zip(reversed(weights), reversed(zetas[:-1])):
                err = np.dot(np.transpose(w), err) * sig_prime(z)
                errors.append(err)

            errors.reverse()

            """
            for err in errors:
                print(err.shape)
                
            print('---------')

            for gb in gradient_b:
                print(gb.shape)

            print('--------------------------')
            quit()
            """

            #dd([ np.dot( errors[0], np.transpose(activations[0]) ).shape ])
             
            #dd([ errors[1].shape, np.transpose(activations[0]).shape ])
            #dd( np.dot(errors[1], np.transpose(activations[0])) )
            gradient_b = [b + e for b, e in zip(gradient_b, errors)]
            
            j = 0
            for a, err in zip(activations, errors):
                gradient_w[j] = gradient_w[j] + (np.dot( err,  np.transpose(a) ))
                j = j + 1
                
            """
            for grad in gradient_w:
                print(grad.shape)
                
            print('-------------')

            for w in weights:
                print(w.shape)
            """
        
        avg = 1 / len(batches)
        for l in range(0, len(layerSizes) - 1):
            weights[l] = weights[l] - learningRate * gradient_w[l] * avg

        for l in range(0, len(layerSizes) - 1):
            biases[l] = biases[l] - learningRate * gradient_b[l] * avg
        #for w in weights:
        #    print(w.shape)

correct = 0;
for a, n in test_data:
    for b, w in zip(biases, weights):
        z = np.dot(w, a) - b
        a = sig(z)
        
    a = softmax(z)
    
    if(a.index(max(a)) == n):
        correct = correct + 1;

accuracy = round(correct / len(test_data), 2)

print("accuracy: " + str(accuracy))