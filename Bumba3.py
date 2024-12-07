# Bumba2, tikai pievienoju vēl "particles", 2 funkcijas : 1. izveido visus objektus, 2. uzzīmē garfiku no objektiem

import numpy as np
from matplotlib import pyplot as plt


def string_to_function(expression):
    """Converts a string to a numpy function"""
    def function(x):
        return eval(expression)
    return np.frompyfunc(function, 1, 1)


resolution : int = 2000

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
        self._x = x
        self._y = y
        self._r = r

    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def r(self):
        return self._r
    

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

def Project(v1 , v2):
    scalar_projection = np.dot(v1, v2)
    v_projection = np.linalg.norm(v2) * scalar_projection

    return v_projection


def makeparticles(x_coord_array):
    radius = (x[-1]-x[0])/(2*len(x))
    particles = []
    for n in x_coord_array:
        particles.append(CollisionParticle(n, defined_function(n), radius))
    return particles

def plotparticles(particles_array):
    fig, ax = plt.subplots()
    for particle in particles_array:
        circle = plt.Circle((particle.x, particle.y), particle.r, color='r')
        ax.add_patch(circle)
        ax.set_aspect('equal')

plotparticles(makeparticles(x))


# plt.plot(x, y)

plt.show()
