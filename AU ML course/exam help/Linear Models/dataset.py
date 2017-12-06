import numpy as np
import matplotlib.pyplot as plt

class TargetFunction():
    def __init__(self, function, string_representation): 
        self.function = function
        self.string_representation = string_representation
        
    def __str__(self):
        return self.string_representation
    

class DataSet:
    
    # Potential features:
    # - Add parameter that allows to decide if bias should be encoded in data.
    # - Add parameter that allows to decide if labels should be {0, 1} or {-1, +1}
    def __init__(self, name, n=100, d=2):
        self.name = name
        if name == "perceptron": 
            self.X, self.y = self.make_classification(n, 2, means = np.array([[2,4], [8,5]]))
        elif name == "pocket":
            self.X, self.y = self.make_classification(n, 2, means = np.array([[3,5], [8,5]]))
        elif name == "linear_regression":
            self.X, self.y, self.target_function = self.make_regression(n, d)
        elif name == "linear_classification":
            pass
        
    def make_regression(self, n, d):
        """ For now assumes d=2, make data normally distributed around line. """
        
        if d == 2:
            # Generate normally distributed noise that displaces points from line. 
            noise_variance = 0.5
            normal_distributed_noise = np.random.normal(loc=0, scale=noise_variance, size=n)

            # Generate random line f(x)=ax+b such that points normally distributed around line will 
            # have high probability of being inside plot. 
            b = np.random.rand(1)*5+2 # Let 'b' be in [4, 6] uniform random
            sign = np.random.choice([-1, +1])
            a = sign* np.random.rand(1)/10*4 # let 'a' be in [-4/10+noise_var, 4/10-noise_var] so all data are in uniform 10,10 box

            target_function = TargetFunction(lambda x: a*x+b, str(round(a[0], 2)) + "*x+" + str(round(b[0], 2)))
            target_function.w = [b, a]

            xs = np.ones((n, 2)) 
            xs[:, 1] = np.random.rand(n)*10 

            ys = target_function.function(xs[:, 1]) + normal_distributed_noise

            return xs, ys, target_function
        else:
            # Generate weight vector
            w = np.random.rand(d)
            
            noise_variance = 0.05
            normal_distributed_noise = np.random.normal(loc=0, scale=noise_variance, size=n)
            
            X = np.random.rand(n, d)
            y = X @ w + normal_distributed_noise
            
            return X, y, None
        
        

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
        
    def plot_regression(self):
        n, d = self.X.shape
        #assert d == 2, "Data needs to be 2d (with bias encoded) to be plotted."
        
        plt.title("Dataset: " + self.name + ", Target Function: " + str(self.target_function))
        plt.xlabel("X dimension of data")
        plt.ylabel("Y dimension of data")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.plot(self.X[:,1], self.y, 'go')
        plt.show()
