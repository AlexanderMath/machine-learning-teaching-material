import matplotlib.pyplot as plt
import numpy as np
import time  
        

def visualize_linreg(self, X, y):
    n, d = X.shape

    self.ax_data.set_title("Linear Regression ")
    self.ax_data.set_xlabel("X dimension of data")
    self.ax_data.set_ylabel("Y dimension of data")

    self.ax_data.plot(X[:,1], y, 'go')

    self.ax_data.set_xlim(0, 10)
    self.ax_data.set_ylim(0, 10)

    # visualize line 
    self.ax_data.plot([0, 10], [self.w[0], self.w[1]*10+self.w[0]], '-r')

    # plot lines for each point to line
    plt.show()

def visualize_linreggd(self, X, y):
    n, d = X.shape

    self.errors.append(self.error(X, y))

    self.ax_data.cla()
    self.ax_data.set_title("Linear Regression (Gradient Descent)")
    self.ax_data.set_xlabel("X dimension of data")
    self.ax_data.set_ylabel("Y dimension of data")
    self.ax_data.set_xlim(0, 10)
    self.ax_data.set_ylim(0, 10)
    self.ax_data.plot(X[:,1], y, 'go')

    self.ax_error.set_title("Average Sum Squared Error")
    self.ax_error.set_xlabel("X dimension of data")
    self.ax_error.set_ylabel("Y dimension of data")
    self.ax_error.plot(range(1, len(self.errors)+1), self.errors, 'b-')

    # visualize line 
    self.ax_data.plot([0, 10], [self.w[0], self.w[1]*10+self.w[0]], '-r')

    # plot lines for each point to line
    self.fig.canvas.draw()

def visualize_linregclass(self, X, y):
    n, d = X.shape

    self.ax_data.set_title("Classification with Linear Regression ")
    self.ax_data.set_xlabel("X dimension of data")
    self.ax_data.set_ylabel("Y dimension of data")

    # plot
    self.ax_data.plot(X[y==-1][:,1], X[y==-1][:,2], 'go')
    self.ax_data.plot(X[y==1][:,1], X[y==1][:,2], 'bx')

    if not self.xlim is None: self.ax_data.set_xlim(self.xlim)
    else: self.xlim = [np.min(X[:,1]), np.max(X[:,1])]
    if not self.ylim is None: self.ax_data.set_ylim(self.ylim)
    else: self.ylim = [np.min(X[:,2]), np.max(X[:,2])]

    # visualize line  
    a = - self.w[1] / self.w[2]
    b = - self.w[0] / self.w[2]

    x0 = a*self.xlim[0]+b
    x1 = a*self.xlim[1]+b

    self.ax_data.plot(self.xlim, [x0, x1], '-r')

    self.ax_data.set_xlim(self.xlim)
    self.ax_data.set_ylim(self.ylim)

    self.fig.canvas.draw()

def visualize_pocket(self, X, y):
    # Draw the best hypothesis soo far with dashed lines. 
    w_ = self.best_w
    
    if np.allclose(w_[2], 0): return
    
    xlim_own = [np.min(X[:,1]), np.max(X[:,1])]
    ylim_own = [np.min(X[:,2]), np.max(X[:,2])]
    self.ax_data.set_xlim(xlim_own)
    self.ax_data.set_ylim(ylim_own)
    
    if not self.xlim is None: 
        self.ax_data.set_xlim(self.xlim)
        xlim_own = self.xlim
    if not self.ylim is None: 
        self.ax_data.set_ylim(self.ylim)
        ylim_own = self.ylim
    
    y0 = (-w_[1]*xlim_own[0] - w_[0]) / w_[2]
    y1 = (-w_[1]*xlim_own[1] - w_[0]) / w_[2]

    self.ax_data.plot(xlim_own, [y0, y1], '--c', ms="8")        
    self.fig.canvas.draw()
    time.sleep(self.sleep)
    
def visualize_logreg_save(self, X, y):
     # Compute error and append to list
    self.errors.append(self.error(X, y))
    self.errors_01.append(self.classification_error(X, y))

    self.ax_error.cla()
    self.ax_error.plot(range(1, len(self.errors) +1), self.errors, '-b')
    self.ax_error.set_xlabel("Iterations")
    self.ax_error.set_ylabel("Error")
    self.ax_error.set_title("Cross Entropy Loss")

    self.ax_error_01.cla()
    self.ax_error_01.plot(range(1, len(self.errors) +1), self.errors_01, '-g')
    self.ax_error_01.set_ylim(0, 1)
    self.ax_error_01.set_xlabel("Iterations")
    self.ax_error_01.set_ylabel("Error")
    self.ax_error_01.set_title("0-1 Classification Error")

    self.ax_data.cla()

    self.ax_data.set_xlabel("X axis of data")
    self.ax_data.set_ylabel("Y axis of data")
    self.ax_data.set_title("Logistic Regression")

    # plot data
    X_0 = X[y==0]
    X_1 = X[y==1]
    self.ax_data.plot(X_0[:,1], X_0[:,2], 'go')
    self.ax_data.plot(X_1[:,1], X_1[:,2], 'bo')

    self.ax_data.set_xlim(0,10)
    self.ax_data.set_ylim(0,10)

    if not self.xlim is None: self.ax_data.set_xlim(self.xlim)
    if not self.ylim is None: self.ax_data.set_ylim(self.ylim)


    # Draw probabilities
    if not self.xlim is None: 
        step_size = 50
        self.xlim = [self.xlim[0]-step_size, self.xlim[1]+step_size]
        self.ylim = [self.ylim[0]-step_size, self.ylim[1]+step_size]

        xdiff = np.abs(self.xlim[0]-self.xlim[1])
        ydiff = np.abs(self.ylim[0]-self.ylim[1])
        grid_points = np.array([(1, i, j) for i in range(self.xlim[0], self.xlim[1], step_size) 
                                        for j in range(self.ylim[0], self.ylim[1], step_size)])
        pred = self.predict(grid_points).reshape((xdiff//step_size, ydiff//step_size))
        xs = np.arange(self.xlim[0], self.xlim[1], step_size)
        ys = np.arange(self.ylim[0], self.ylim[1], step_size)
        c = self.ax_data.contourf(xs, ys, pred.T, vmin=0, vmax=1, cmap="RdBu")

        self.xlim = [self.xlim[0]+step_size, self.xlim[1]-step_size]
        self.ylim = [self.ylim[0]+step_size, self.ylim[1]-step_size]

    else: 
        grid_points = np.array([(1, i, j) for i in range(11) for j in range(11)])
        pred = self.predict(grid_points).reshape((11, 11))
        c = self.ax_data.contourf(np.arange(11), np.arange(11), pred, vmin=0, vmax=1, cmap="RdBu")

    # draw proababilities
    self.fig.canvas.draw()
    time.sleep(self.sleep)
    
    
    
def visualize_logreg(self, X, y):
      # Compute error and append to list
    self.errors.append(self.error(X, y))
    self.errors_01.append(self.classification_error(X, y))

    self.ax_error.cla()
    self.ax_error.plot(range(1, len(self.errors) +1), self.errors, '-b')
    self.ax_error.set_xlabel("Iterations")
    self.ax_error.set_ylabel("Error")
    self.ax_error.set_title("Cross Entropy Loss")

    self.ax_error_01.cla()
    self.ax_error_01.plot(range(1, len(self.errors) +1), self.errors_01, '-g')
    self.ax_error_01.set_ylim(0, 1)
    self.ax_error_01.set_xlabel("Iterations")
    self.ax_error_01.set_ylabel("Error")
    self.ax_error_01.set_title("0-1 Classification Error")

    self.ax_data.cla()

    self.ax_data.set_xlabel("X axis of data")
    self.ax_data.set_ylabel("Y axis of data")
    self.ax_data.set_title("Logistic Regression")

    # plot data
    X_0 = X[y==0]
    X_1 = X[y==1]
    self.ax_data.plot(X_0[:,1], X_0[:,2], 'go')
    self.ax_data.plot(X_1[:,1], X_1[:,2], 'bo')
    
    
    
    self.ax_data.set_xlim(0,10)
    self.ax_data.set_ylim(0,10)

    if not self.xlim is None: self.ax_data.set_xlim(self.xlim)
    if not self.ylim is None: self.ax_data.set_ylim(self.ylim)

        
    # Draw probabilities
    if not self.xlim is None: 
        xdiff = np.abs(self.xlim[0]-self.xlim[1])
        ydiff = np.abs(self.ylim[0]-self.ylim[1])
        
        grid_points = np.array([(1, i, j) for i in range(xdiff) for j in range(ydiff)])
        pred = self.predict(grid_points).reshape((xdiff, ydiff))
        c = self.ax_data.contourf(np.arange(xdiff), np.arange(ydiff), pred, vmin=0, vmax=1, cmap="RdBu")
    else: 
        grid_points = np.array([(1, i, j) for i in range(11) for j in range(11)])
        pred = self.predict(grid_points).reshape((11, 11))
        c = self.ax_data.contourf(np.arange(11), np.arange(11), pred, vmin=0, vmax=1, cmap="RdBu")

    # draw proababilities
    self.fig.canvas.draw()
    time.sleep(self.sleep)
        
def init_perceptron(self):
    self.fig, (self.ax_data, self.ax_errors) = plt.subplots(1, 2, figsize=(11, 5))
    self.errors = []
    self.ax_data.set_title("Perceptron Learning Algorithm")
    self.ax_data.set_xlabel("X dimension of data")
    self.ax_data.set_ylabel("Y dimension of data")
    self.ax_errors.set_title("Training error at each iteration")
    self.ax_errors.set_xlabel("Iterations")
    self.ax_errors.set_ylabel("Training error")

def visualize_perceptron(self, X, y, subclass):

    """ Visualizes a step of the Perceptron learning algorithm. Assumes the data
        is 2-dimensional (with bias encoded so actually 3 dimensional). Throws an exception 
        if this is not the case. 

         Parameters
        ----------
        X:    Matrix with shape (n, 2) with data point x_i as the i'th row.   
        y:    Array with shape (n, ) with label y_i on the i'th entry.         
    """
    # Check dimension of data. 
    n, d = X.shape
    assert d == 3, "Data should be two dimensional with bias encoded. "

    # Split data into classes
    X_class_0 = X[y==-1]
    X_class_1 = X[y==1]


    # Draw the different classes of data
    self.ax_data.cla()
    
    plot_range = [np.minimum(np.min(X[:,1]), np.min(X[:,2])), 
                  np.maximum(np.max(X[:,1]), np.max(X[:,2]))]

    
    self.ax_data.set_xlim(plot_range[0], plot_range[1])
    self.ax_data.set_ylim(plot_range[0], plot_range[1])

    self.ax_data.plot(X_class_0[:,1], X_class_0[:,2], 'go')
    self.ax_data.plot(X_class_1[:,1], X_class_1[:,2], 'bo')


    # Draw the hyperplane
    w = self.w
    
    if np.allclose(w[2], 0): return
    
    y0 = (-w[1]*plot_range[0] - w[0]) / w[2]
    y1 = (-w[1]*plot_range[1] - w[0]) / w[2]
    
    self.ax_data.plot(plot_range, [y0, y1], '-c', ms="8")


    # Highlight missclassified points, and highlight the chosen one. 
    # Predict the class of each data point. 
    predictions = self.predict(X)

    # Get the number of misclassified points
    misclassified_count = sum(predictions != y)

    # Proceed only if there are miss classified points. 
    if misclassified_count != 0: 

        # Filter out the points where predictions disagree with labels. 
        misclassified_points = X[predictions != y]
        misclassified_labels = y[predictions != y]

        # Mark all the miss classified points by a yellow cross 'yx'. 
        self.ax_data.plot(misclassified_points[:, 1], misclassified_points[:, 2], 'ro', ms=4)

        # Mark the chosen miss classified point by a large red dot. 
        misclassified_point = misclassified_points[0]
        self.ax_data.plot(misclassified_point[1], misclassified_point[2], 'co', ms=10)

        # Draw an arrow from hyperplane towards this point. 
        # hypothesis is f(x)= -w[1]/w[2] * x - w[0]/w[2]
        # we want the arrow to be orthogonal to f(x), so if we think of it
        # as a line the slopes both lines should give -1.  
        # The intercept of the line is now given b=y_mc + x_mc*w[1]/w[2].
        # Then we want to compute the intersection point of the two lines and we are done. 
        x_mc = misclassified_point[1]
        y_mc = misclassified_point[2]

        f_a = -w[1]/w[2]
        f_b = -w[0]/w[2]

        g_a = - 1 / f_a     # g_a*f_a = -1 => g_a = - 1 / f_a 
        g_b = y_mc - g_a * x_mc # y = a*x + b => b = y - a*x

        x_intercept = (g_b - f_b) / (f_a - g_a)
        y_intercept = x_intercept * f_a + f_b

        self.ax_data.plot(x_intercept, y_intercept, 'co', ms=8)
        self.ax_data.arrow(x_intercept, y_intercept, x_mc - x_intercept, y_mc - y_intercept, 
                           head_width=0.8, head_length=0.8, fc='k', ec='k')

        # Draw error of iterations
        self.ax_errors.cla()
        self.ax_errors.set_ylim(0, 1)
        self.errors.append(self.classification_error(X, y))
        self.ax_errors.plot(range(1, len(self.errors)+1), self.errors, '-g')

        # Titles and axis labels
        self.ax_data.set_title("Perceptron Learning Algorithm")
        self.ax_data.set_xlabel("X dimension of data")
        self.ax_data.set_ylabel("Y dimension of data")
        self.ax_errors.set_title("Training error at each iteration")
        self.ax_errors.set_xlabel("Iterations")
        self.ax_errors.set_ylabel("Training error")

        # Update plot
        if not subclass: 
            self.fig.canvas.draw()
            time.sleep(self.sleep)