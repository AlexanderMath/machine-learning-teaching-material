import numpy as np
import matplotlib.pyplot as plt

class DataSet:
    
    # Potential features:
    # - Add parameter that allows to decide if bias should be encoded in data.
    # - Add parameter that allows to decide if labels should be {0, 1} or {-1, +1}
    def __init__(self, name):
        self.name = name
        if name == "perceptron": 
            self.X, self.y = self.make_classification(100, 2, means = np.array([[2,4], [8,5]]))
        elif name == "pocket":
            self.X, self.y = self.make_classification(100, 2, means = np.array([[3,5], [8,5]]))

    def make_classification(self, n, d, means=None, num_classes=2, linear_seperable=False):
        """ Creates data for a 'num_classes' classification problem. All points are generated in a 
        cube [0, 2]^d. Each class is generated as a normal distribution N(µ, 1) around a 
        randomly generated mean. 

        """
        # Generate num_classes means
        if means is None: 
            means = np.random.rand(num_classes, d)*10
        
        # Initialize data matrix and labels array
        # Encode 1's in first dimension
        X = np.ones((n, d+1))
        y = np.zeros(n, dtype=np.int32)

        for i in range(n):
            y[i] = np.random.choice(num_classes)
            X[i, 1:d+1] = np.random.normal(loc=means[y[i]], scale=0.8)

        # Have labels be {-1, +1}
        y = y*2-1
            
        return X, y
    
    
    def plot(self):
        """ Assumes the data is 2d and plots it. Throws exception if data isn't 2d (with bias encoded). 
        
        """
        n, d = self.X.shape
        assert d == 3, "Data needs to be 2d (with bias encoded) to be plotted."
        
        X_class_0 = self.X[self.y == -1]
        X_class_1 = self.X[self.y == 1]

        plt.title("Dataset: " + self.name)
        plt.xlabel("X dimension of data")
        plt.ylabel("Y dimension of data")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.plot(X_class_0[:,1], X_class_0[:,2], 'go')
        plt.plot(X_class_1[:,1], X_class_1[:,2], 'bo')
        plt.show()

        