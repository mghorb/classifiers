
import matplotlib.pyplot as plt
import numpy as np

def question1(columns):
    ###### QUESTION 1A ######
    # initializes a 1000xc matrix random numbers each number ranging between 0 and 1
    array = np.random.rand(1000, columns)

    ###### QUESTION 1B ######
    # (runtime ~20s) finding correlation coeff matrix, exact to:
    # corrcoefs = np.corrcoef(array)

    # initlialize an empty coefficient correlation matrix
    corrcoefs = np.zeros([1000, 1000])
    # correlation matrix is symmetrical, calculating for half the triangle cuts down run time
    for x in range(1000):
        for y in range(x + 1):
            if x == y:
                # diagnol is all 1s, calculatin corr. coef. of row against itsself
                corrcoefs[x, y] = 1
            else:
                # calculate x & y bars
                xbar = np.mean(array[x])
                ybar = np.mean(array[y])
                # calculate r value
                r = np.sum((array[x] - xbar) * (array[y] - ybar)) / (
                            np.sqrt(np.sum(np.power((array[x] - xbar), 2))) * np.sqrt(
                        np.sum(np.power((array[y] - ybar), 2))))
                # symmetrical matrix , put it in symmetrical places
                corrcoefs[x, y] = r
                corrcoefs[y, x] = r

    ###### QUESTION 1C ######
    # gets lower triangle of the matrix, without diagnol
    corrlt = np.tril(corrcoefs, k=-1)
    # masked all values equal to 0
    corrlt = np.ma.masked_equal(corrlt, 0)
    # returns all non-masked values as 1D array
    corrlt = np.ma.compressed(corrlt)

    # find number of elements above 0.75
    probmore = (corrlt > 0.75).sum()
    # find number of elements below -0.75
    probless = (corrlt < -0.75).sum()
    # find probablity of those obtaining those elements
    prob = (probless + probmore) / len(corrlt)

    # creates & shows histogram
    plt.title(str(columns) + " Element Histogram P(r < -0.75, r > 75) = " + str("{:.5f}".format(prob)))
    plt.hist(corrlt, bins=100)
    plt.show()
    
    
print("Question 1 (50)")
question1(50)
print("\n\nQuestion 1 (10)")
question1(10)