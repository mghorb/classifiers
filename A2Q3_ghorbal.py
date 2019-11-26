"""
Name: Manel Ghorbal
Assignment 2 Question 3
"""

from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import numpy as np
import math

# classifies the class into binary 0 and 1
def vector_class(target):
    # calc median of data
    median = np.median(target)
    # initialize vector
    vector = np.zeros(len(target))
    for i in range(len(target)):
        if target[i] >= median:
            # if value larger than median = 1, else already equal to 0
            vector[i] = 1
    return vector


# xi = instance attribute value, classx = instances of class
def prob(xi, classx):
    return math.sqrt(1/(2 * math.pi * math.pow(np.std(classx), 2))) * math.exp(-math.pow(xi - np.mean(classx), 2)/(2 * math.pow(np.std(classx), 2)))

# Calculates GNB
def GNB(instance, classp, classn):
    total = len(classp) + len(classn)
    # P(class0) = # instances of class 0 / # of data instances
    probcp = len(classp) / total
    # P(class1) = # instances of class 1 / # of data instances
    probcn = len(classn) / total

    # calc probs
    ppos = prob(instance[0], classp[:, 0]) * prob(instance[1], classp[:, 1]) * prob(instance[2], classp[:, 2]) * probcp
    pneg = prob(instance[0], classn[:, 0]) * prob(instance[1], classn[:, 1]) * prob(instance[2], classn[:, 2]) * probcn
    
    return (ppos / (ppos + pneg))


# calculate the prediction classes & perceptron
def percetron(tdata, classes):    
    # set all weights to zero
    weights = np.zeros([4,1]) 
    
    # until all instances in training set are correctly classified or max # iterations is reached
    for i in range(0,iterations):
        # for each instance 
        for instance in range(0,tdata.shape[0]): 
            # calculate the prediction 
            prediction = np.dot(tdata[instance, :], weights)
            if prediction>0: 
                pclass=1
            else:
                pclass=0
            if pclass != classes[instance]: 
                # if the instance belongs to first class, add it to weight vector
                if classes[instance] == 1: 
                    weights = weights + np.expand_dims(tdata[instance, :],1)
                # otherwise subtract it from weight vector
                else: 
                    weights = weights - np.expand_dims(tdata[instance, :],1)
        
    return np.dot(tdata,weights)

# load data
d = load_linnerud()

# seperate values in class 0/1 (d.data[:, 0] = chinups)
classes = vector_class(d.data[:, 0])
# concatonate classified vector to attribute values
chins = np.concatenate((d.target, classes[np.newaxis].T), axis=1)
# create two sub datasets based on classes
class0 = chins[chins[:, 3] == 0]
class1 = chins[chins[:, 3] == 1]

""" ####### GAUSSIAN NAIVE BAYES ####### """
# create a text file to write results 
f= open("q3GNB.txt","w+")
f.write("Question 3 GNB Results\nManel Ghorbal\n")

# for each instance, calculate GNB & write it into file
for i in range(len(chins)):  
    f.write("\nP(chin ups = 1 | instance %d) = " % (i+1))
    print("P(chin ups = 1 | instance %d) = %f Instance class: %d" % (i+1, GNB(d.target[i], class1, class0), chins[i, 3]))
    
    f.write(str("{:.5f}".format(GNB(d.target[i], class1, class0))))
    f.write("\n")
f.close()
    
""" ####### Perceptron ####### """

# number of iterations
iterations = 1000

# add a column of zeros to dataset for the intercept
tdata = np.zeros([20,4])
tdata[:,0:3] = d.target
tdata[:,3] = 1

# calculate predictions
predictions = percetron(tdata, classes)

# write predictions in a text file
g = open("q3Perceptron.txt","w+")
g.write("Question 3 Perceptron Results\nManel Ghorbal\n")
for i in range(len(tdata)):
    g.write("Prediction Value instance %d = " % (i+1))
    g.write(str(predictions[i, 0]))
    g.write("\n")
g.close()

plt.title("Perceptron Predictions iterations = " + str(iterations))
#plot prefdiction values
plt.plot(predictions,'o')
# plot perceptron line (on x axis)
plt.plot([0,20], [0,0])
    
    