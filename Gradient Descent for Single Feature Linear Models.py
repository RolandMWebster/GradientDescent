# IMPORT PACKAGES =============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GENERATE OBSERVATIONS =======================================================
def observationsGenerator(n):
    """ Generates n points from a function y = a0 + a1*x + e(0,1) and plots
    the points. """
    # Generate random x values
    x_values = np.random.uniform(0,100, n)
        
    # Generate random coefficients a0 and a1:
    a0 = np.random.uniform(0, 100)
    a1 = np.random.uniform(0, 10)
        
    # Generate error values
    errors = np.random.normal(0, 100, n)
    
    # Generate random y values
    y_values = a0 + (x_values * a1) + errors
    
    # plot (x,y)
    plt.scatter(x_values, y_values)
    
    output = pd.DataFrame({'x':x_values, 'y':y_values})
    
    return(output)


# GRADIENT DESCENT ============================================================

# Create a function to carry out gradient descent for single feature linear models.
def linearGradientDescent(data, alpha = 0.0001, batches = 10, a0 = 0, a1 = 0):
    """ Performs gradient descent for lienar models using a learning rate alpha.
    The number of iterations is provided by the user via the batches argument."""

    # Assign x, y, and number of observations m:
    x = data['x']
    y = data['y']    
    m = data.shape[0]
        
    # Get our x points for the line plot
    x_points = np.linspace(min(x), max(x), 2)

    # Plot our starting function (using the arguments given in the function
    # as the coefficients)
    plt.plot(x_points, a0 + a1*x_points, color = "blue")

    # Loop through our user provided number of batches
    for i in range(batches):
        
        # Create temp0
        temp0 = a0 - (alpha / m) * sum((a0 + (x * a1)) - y)
        
        # Create temp1
        temp1 = a1 - (alpha / m) * sum(((a0 + (x * a1)) - y) * x)
        
        # Update a0
        a0 = temp0
        
        # Update a1
        a1 = temp1
     
        # Plot our function after gradient descent:
        
        # Plot our final function as a red line    
        if i == (batches - 1):
            plt.plot(x_points, a0 + a1*x_points, color = "red")
        # Plot intermediate steps
        else:
            plt.plot(x_points, a0 + a1*x_points, color = "grey", linestyle = "--")
            
    # Add our observations to the plot
    plt.scatter(x, y, color = "black")
    
    # Store our parameters in a DataFrame
    output = pd.DataFrame({'a0':[a0], 'a1':[a1]})
    
    return(output)
    

# TEST IT OUT =================================================================
observations = observationsGenerator(100)    
results = linearGradientDescent(observations)
    
