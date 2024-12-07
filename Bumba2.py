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
    

class Ball:
    # pass all the 
    def __init__(self, r = [0.5,0.5]) -> None:
        self.x = r[0]
        self.y = r[1]

    @property
    def x(self):
        return self.r[0]
    @property
    def y(self):
        return self.r[1]
    @property
    def r(self):
        return self.r
    
    @x.setter
    def x(self, value):
        self.r[0] = value
    def y(self, value):
        self.r[1] = value

    def Draw(self,ax):
        circle = plt.Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle


class Constants():
    def __init__(self, g = (0, -9.81), m : float = 1, nu : float = 0.15) -> None:
        self.g = g
        self.m = m
        self.nu = nu

    @property
    def g(self):
        return self.g
    @property
    def m(self):
        return self.m
    @property
    def nu(self):
        return self.nu
        

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


## use for animation debugging
# fig = plt.gcf()
# fig.canvas.manager.set_window_title('Test')
## OR
# plt.title( 'Colored Circle' )


for i in range(0, 4):
    ax = plt.gca()
    Ball = Ball(mass=1, startPos=(0.5,0.5),radius=0.04)
    Ball.draw(ax=ax)
    Ball.advance(dt=0.01, accel=9.81)
    
    ax.set_aspect( 1 ) # set aspect ratio
    # ax.add_artist( Drawing_colored_circle )
    plt.title( 'Colored Circle' )

    plt.plot(x,y)
    plt.show()

# plt.plot(x, y, label="Grafiks")
# plt.legend()

# plt.show()
