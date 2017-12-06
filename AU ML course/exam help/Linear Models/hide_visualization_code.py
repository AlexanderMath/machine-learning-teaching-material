import matplotlib.pyplot as plt
import time


def visualize_pocket(self, X, y):
    # Draw the best hypothesis soo far with dashed lines. 
    w_ = self.best_w
    y0 = (-w_[1]*0 - w_[0]) / w_[2]
    y1 = (-w_[1]*10 - w_[0]) / w_[2]
    self.ax_data.plot([0, 10], [y0, y1], '--c', ms="8")        
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
    self.ax_data.set_xlim(0, 10)
    self.ax_data.set_ylim(0, 10)

    self.ax_data.plot(X_class_0[:,1], X_class_0[:,2], 'go')
    self.ax_data.plot(X_class_1[:,1], X_class_1[:,2], 'bo')


    # Draw the hyperplane
    w = self.w
    y0 = (-w[1]*0 - w[0]) / w[2]
    y1 = (-w[1]*10 - w[0]) / w[2]
    self.ax_data.plot([0, 10], [y0, y1], '-c', ms="8")


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
        self.errors.append(self.error(X, y))
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