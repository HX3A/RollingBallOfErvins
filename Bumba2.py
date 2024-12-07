import numpy as np
from matplotlib import pyplot as plt


def string_to_function(expression):
    """Converts a string to a numpy function"""
    def function(x):
        return eval(expression)
    return np.frompyfunc(function, 1, 1)


resolution : int = 100

# function equation
# formula : str = "np.tanh(x)"
formula : str = "1*x**2-2*x+1"


defined_function = string_to_function(formula)

# x = np.array((1,2),0.01)
from_x = 0
to_x = 1.5


x = np.linspace(from_x, to_x, resolution)
y = defined_function(x)

class CollisionParticle():

    # create a lot of small circles 
    def __init__(self, x : float, y : float, r : float) -> None:
        self.x = x
        self.y = y
        self.r = r

    @property
    def x(self):
        return self.x
    @property
    def y(self):
        return self.y
    @property
    def r(self):
        return self.r
    

class Ball():
    # pass all the 
    def __init__(self, x : float, y : float) -> None:
        self.x = x
        self.y = y
        pass

    @property
    def x(self):
        return self.x
    @property
    def y(self):
        return self.y


def TangentToParticle(p1 : CollisionParticle, ball : Ball):
    # pass the CLOSEST particle in here

    v1 = np.array(ball.x, ball.y) - np.array(p1.x, p1.y)
    v1_x = v1[0]
    v1_y = v1[1]

    #The direction may be flipped, just * by -1 if thats the case
    t1 = np.array([-v1_y, v1_x]) 
     
    return t1 

def Project(v1 , v2 ):
    scalar_projection = np.dot(v1, v2)
    v_projection = np.linalg.norm(v2) * scalar_projection

    return v_projection


plt.plot(x, y)

plt.show()
