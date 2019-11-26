"""
Name: Manel Ghorbal
Assignment 2 Question 2
"""
import numpy as np 
from numpy.linalg import inv 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_linnerud


def question2(axis, xdata, ydata, xlabel, ylabel, title):
    # set labels of table
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    
    # plot the attribute values & the 
    axis.plot(xdata, ydata, 'o')
    
    ### calculate linear least squares line ###
    # gives the same results as:
    # m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
    # axis.plot(x, m*x + c, 'r', label='Fitted line')
    
    # add a column of ones to the x data for the intercept
    x = np.zeros([len(xdata),2])
    x[:,0] = 1
    x[:,1] = xdata 
    
    # expand dimensions of y data for the dot product
    y = np.expand_dims(ydata, axis=1)
    
    # compute the weights
    W = np.dot(np.dot(inv((np.dot(x.T, x))), x.T), y)
    
    # plot points of data
    axis.plot(x[:,1], y, 'o')
    # plot the least square line 
    axis.plot(x[:,1], np.dot(x,W))

    # calculate correlation coefficient
    r = (len(xdata) * np.sum(x[:,1] * ydata) - np.sum(xdata) * np.sum(ydata)) / (len(xdata) * np.sum(np.power(xdata, 2)) - np.power(np.sum(xdata), 2))
    # calculate intercept
    b = (np.sum(ydata) - (r * np.sum(xdata))) / len(xdata)
    
    # set title for graph
    axis.set_title(title + " Slope: " + str("{:.4f}".format(r)) + " Intercept: " + str("{:.4f}".format(b)))


# load data
d = load_linnerud()

# feature_names = ['Chins', 'Situps', 'Jumps']
chins = d.data[:, 0]
situps = d.data[:, 1]
jumps = d.data[:, 2]

# target_names = ['Weight', 'Waist', 'Pulse']
weight = d.target[:, 0]
waist = d.target[:, 1]
pulse = d.target[:, 2]

# create subplot figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 3)

# increase size of figure
fig.set_size_inches(16, 9)

print("Question 2")
# find line of best fit for each set
question2(ax1[0], weight, chins, "weight", "chinups", "Weight vs Chinups")
question2(ax1[1], weight, situps, "weight", "situps", "Weight vs Situps")
question2(ax1[2], weight, jumps, "weight", "jumps", "Weight vs Jumps")
question2(ax2[0], waist, chins, "waist", "chinups", "Waist vs Chinups")
question2(ax2[1], waist, situps, "waist", "situps", "Waist vs Situps")
question2(ax2[2], waist, jumps, "waist", "jumps", "Waist vs Jumps")
question2(ax3[0], pulse, chins, "pulse", "chinups", "Pulse vs Chinups")
question2(ax3[1], pulse, situps, "pulse", "situps", "Pulse vs Situps")
question2(ax3[2], pulse, jumps, "pulse", "jumps", "Pulse vs Jumps")

plt.tight_layout()

