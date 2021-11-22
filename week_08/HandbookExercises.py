import numpy as np

import array

L = list(range(10))
A = array.array('i', L)

np.array([1,2,3,4,5])

# Making a two dimensional array
np.array([range(i, i + 3) for i in [2, 4, 6]])

# length 10 integer array of 0's
np.zeros(10, dtype = int)

# 3x5 floating-point array of 1's
np.ones((3, 5), dtype = float)

# 3x5 array of 1.14
np.full((3, 5), 3.14)

# Create an array of linear sequence, starting a 0, ending at 20 stepping by 2
np.arange(0, 20, 2)

# Array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

#create a 3x3 matrix of uniformly distributed values with mean 0, and std dev 1
np.random.normal(0, 1, (3, 3))

# create a 3x3 identity matrix
np.eye(3)

# you can specify data type (int vs float fx)
np.zeros(10, dtype = 'int16') #defined
np.zeros(10) # not defined

### NumPy array attributed ###

# Create three data frames
np.random.seed(0)
x1 = np.random.randint(10, size=6) # one dimensional
x2 = np.random.randint(10, size=(3, 4)) # two dimensional
x3 = np.random.randint(10, size=(3, 4, 5)) # three dimensional

# Getting to know a variable a little
print("x3 ndim: ", x3.ndim) # Number of dimensions
print("x3 shape:", x3.shape) # Shape of data frame
print("x3 size: ", x3.size) # total size of array (3*4*5)
print("x3 data type: ", x3.dtype) # type of data

#### Accessing single elements of array ####
# In a one dimensional
x1[0] # first element in array

# in a two dimensional
x2[0,0] # row 1, column 1

# modify values inside array
x2[0,0] = 12
x2

### Array slicing ###
# slice syntax: x[start:stop:step] - default is start=0, stop = size of dimension, step = 1.

x = np.arange(10)

#A potentially confusing case is when the step value is negative. In this case, the defaults for start and stop are swapped. This becomes a convenient way to reverse an array:
In[22]: x[::-1] # all elements, reversed
Out[22]: array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) In[23]: x[5::-2] # reversed every other from index 5 Out[23]: array([5, 3, 1])

print(x2[:,0]) # first column of x2
print(x2[0, :]) # first row of x2

### Reshaping arrays ###
# Put numbers 1-9 in a 3x3 grid:
grid = np.arange(1, 10).reshape((3, 3))
print(grid)

### Combine two (or more) arrays: concatenate() ###
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

np.concatenate([x, y])

# Concatenate can also be done on two dimensional scale
grid = np.array([[1, 2, 3],
                [4, 5, 6]])

np.concatenate([grid,grid]) #concatinate along 1st axis
np.concatenate([grid,grid], axis = 1) #concatinate along 2nd axis


######## LOOPING ###########

np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range (len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1, 10, size=5)


compute_reciprocals(values)

