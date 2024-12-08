# Bumba2, tikai pievienoju vēl "particles", 2 funkcijas : 1. izveido visus objektus, 2. uzzīmē garfiku no objektiem
# Vēl pievienoju divus veidus ātruma aprēķināšanai - ar enerģijas zudumiem un bez. Pievienoju gravitācijas funkciju. Un vēl Collision detection Lite, vajag viņu uztaisīt tā, lai nepārbauda visiem punktiem, bet daļai.
# Klassēs visur pieliku self._x (underscore), jo savādāk nestrādāja

import numpy as np
from matplotlib import pyplot as plt


def string_to_function(expression):
    """Converts a string to a numpy function"""
    def function(x):
        return eval(expression)
    return np.frompyfunc(function, 1, 1)


resolution : int = 200

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
    def __init__(self, x : float, y : float, r : float) -> None:
        self._x = x
        self._y = y
        self.r = r

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, new_x : float):
        self._x = new_x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, new_y : float):
        self._y = new_y

    def draw(self, ax):
        circle = plt.Circle((self.x, self.y), radius=self.r, color='b')
        ax.add_patch(circle)
        return circle
    
    # for scalar -> Positive goues upwards
    def advance(self, dt, accel):
        """Advance the Ball's position forward in time by dt."""
        displacement = ((accel*(dt**2)) / 2)
        # self.r += self.v * dt
        # self.x += displacement[0]
        self.y += displacement
    # for Vector -> Positive goues upwards
    def advance(self, dt, accel : np.ndarray):
        """Advance the Ball's position forward in time by dt."""
        displacement = ((accel*(dt**2)) / 2)
        # self.r += self.v * dt
        # self.x += displacement[0]
        self.x += displacement[0]
        self.y += displacement[1]


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




def collision_detect(ball : Ball, particles_array):
    for particle in particles_array:
        distance = np.sqrt((ball.x - particle.x)**2 + (ball.y - particle.y)**2)
        if distance < 2 * ball.r:
            return True
        else:
            return False

# teoretiski tas pats kaa tev, bet ar np.hypot funkciju
# def collision_detect(ball : Ball, particles_array):
#     for particle in particles_array:
#         distance = np.hypot(np.array(ball.x, ball.y),np.array(particle.x,particle.y))
#         if distance < ball.r:
#             return True
#         else:
#             return False

max_frames = 4

ball = Ball( x = 0.5, y = 0.5, r=0.04)

particleArray = makeparticles(x)
plotparticles(particleArray)

for i in range(0, max_frames):

    ax = plt.gca()
    ax.set_aspect( 1 ) # set aspect ratio

    ball.draw(ax=ax)
    ball.advance(dt=0.1, accel=np.array([-10,-10]))

    if collision_detect(ball,particleArray): # nestrada, nezinu kapeec
        print("Warning! Collision Imminent!")
    
    # ax.add_artist( Drawing_colored_circle )
    plt.title( f'Colored Circle Frame : {i}' )
    plt.plot(x,y)
    plt.show()

# plt.plot(x, y, label="Grafiks")
# plt.legend()

plt.show()
