# Bumba2, tikai pievienoju vēl "particles", 2 funkcijas : 1. izveido visus objektus, 2. uzzīmē garfiku no objektiem
# Vēl pievienoju divus veidus ātruma aprēķināšanai - ar enerģijas zudumiem un bez. Pievienoju gravitācijas funkciju. Un vēl Collision detection Lite, vajag viņu uztaisīt tā, lai nepārbauda visiem punktiem, bet daļai.
# Klassēs visur pieliku self.x (underscore), jo savādāk nestrādāja

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
formula : str = "1*x**2-2*x+1.1"


defined_function = string_to_function(formula)

# x = np.array((1,2),0.01)
fromx = 0
tox = 1.5


x = np.linspace(fromx, tox, resolution)
y = defined_function(x)

class CollisionParticle():

    # create a lot of small circles 
    def __init__(self, x : float, y : float, r : float) -> None:
        self.x = x
        self.y = y
        self._r = r

    @property
    def x(self):
        return self.x
    @property
    def y(self):
        return self.y
    @property
    def r(self):
        return self._r
    

class Ball():
    # pass all the 
    def __init__(self, x : float, y : float, r : float) -> None:
        self.x = x
        self.y = y
        self.r = r

    @property
    def x(self):
        return self.x

    @x.setter
    def x(self, newx : float):
        self.x = newx

    @property
    def y(self):
        return self.y

    @y.setter
    def y(self, newy : float):
        self.y = newy

    def draw(self, ax):
        circle = plt.Circle((self.x, self.y), radius=self.r, color='b')
        ax.add_patch(circle)
        return circle
    
    # for scalar -> Positive goues upwards
    def advance(self, dt, accel : float):
        """Advance the Ball's position forward in time by dt."""
        displacement = ((accel*(dt**2)) / 2)
        # self.r += self.v * dt
        # self.x += displacement[0]
        self.y += displacement
    # for Vector -> Positive goues upwards
    def advance(self, dt, accel : np.ndarray):
        """Advance the Ball's position forward in time by dt."""
        displacement = ((accel*(dt**2)) / 2)

        self.x += displacement[0]
        self.y += displacement[1]


def TangentToParticle(p1 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here

    v1 : np.ndarray = np.array([ball.x, ball.y]) - np.array([p1.x, p1.y])
    v1x = v1[0]
    v1y = v1[1]

    #The direction may be flipped, just * by -1 if thats the case
    t1 : np.ndarray = np.array([-v1y, v1x])
    print(t1, " - perpendicular vector") 
     
    return np.array([-v1y, v1x])  

def Normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def Project(v1 , v2):
    scalar_projection = np.dot(v2, v1)

    v_projection = Normalize(v2) * scalar_projection * 10

    print(Normalize(v2), "Project insides", v_projection, "v projection")

    return v_projection


def makeparticles(x_coord_array):
    radius = (x[-1]-x[0])/(2*len(x))
    particles = []
    for n in x_coord_array:
        particles.append(CollisionParticle(n, np.float64(defined_function(n)), radius))
    return particles

def plotparticles(particles_array):
    fig, ax = plt.subplots()
    for particle in particles_array:
        circle = plt.Circle((particle.x, particle.y), particle.r, color='r')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        # print((particle.x, particle.y))


# def collision_detect(ball : Ball, particles_array):
#     for particle in particles_array:
#         distance = np.sqrt(np.square(ball.x - particle.x) + np.square(ball.y - particle.y))
#         # print(index,distance)
#         if distance < ball.r + particle.r:
#             print("True")
#             return True, particle
        
#     return False, None

def collision_detect(ball : Ball, particles_array):
    collision_particles = []
    for particle in particles_array:
        if particle.x <= ball.x + ball._r and particle.x >= ball.x - ball._r:

            vx = ball.x - particle.x
            vy = ball.y - particle.y
            distance = np.sqrt(np.square(vx) + np.square(vy))
            
            if distance < ball._r + particle._r:
                p = [particle, distance]
                collision_particles.append(p)

    if len(collision_particles) == 0:      
        return False, None, None
    
    closest = collision_particles[0]
    
    for i in range(1, len(collision_particles)):
        if closest[1] > collision_particles[i][1]:
            closest = collision_particles[i]
    distance = closest[1]
    closest = closest[0]

    vx = ball.x - closest.x
    vy = ball.y - closest.y
    displacement_value = ball._r + closest._r - distance
    theta = np.arctan(vy/vx)
    x_displacement = np.cos(theta) * displacement_value
    y_displacement = np.sin(theta) * displacement_value
    ball.x += x_displacement
    ball.y += y_displacement

    particle2 = particleArray[particleArray.index(closest) + 1]

    return True, closest, particle2

max_frames = 8

ball = Ball( x = 0.5, y = 0.5, r=0.04)

particleArray = makeparticles(x)
plotparticles(particleArray)
# print(type(particleArray))

for i in range(0, max_frames):

    ax = plt.gca()
    ax.set_aspect( 1 ) # set aspect ratio

    ball.draw(ax=ax)

    isCollifing, CollidingWithP = collision_detect(ball, np.array(particleArray))

    if isCollifing: # nestrada, nezinu kapeec
        # accelProjection = Project( TangentToParticle(CollidingWithP, ball), np.array([0, -10]))
        accelProjection = Project( np.array([0, -10]), TangentToParticle(CollidingWithP, ball))
        # print(accelProjection, "- Projection")

        ball.advance(dt=0.1, accel=np.array(accelProjection))
        
    else:
        # print("Warning! Collision Imminent!")
        ball.advance(dt=0.1, accel=np.array([0,-10]))
    
    # ax.add_artist( Drawing_colored_circle )
    plt.title( f'Colored Circle Frame : {i}' )
    plt.plot(x,y)
    plt.show()

plt.show()
