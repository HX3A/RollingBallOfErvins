import numpy as np
from matplotlib import pyplot as plt
import imageio
import os, shutil

# Making a folder to save frames in it.
isExist = os.path.exists('Frames')
if not isExist:
    os.makedirs('Frames')
else:
    print('Directory "Frames" already exists')


# Converts a string to a numpy function
def string_to_function(expression):
    def function(x):
        return eval(expression)
    return np.frompyfunc(function, 1, 1)


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
    def __init__(self, x : float, y : float, r : float) -> None:
        self._x = x
        self._y = y
        self._r = r

    @property
    def r(self):
        return self._r

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
        circle = plt.Circle((self._x, self._y), radius=self._r, color='b')
        ax.add_patch(circle)
        return circle
    
    # for Vector -> Positive goues upwards
    def advance(self, dt, accel : np.ndarray):
        """Advance the Ball's position forward in time by dt."""
        displacement = ((accel*(dt**2)) / 2)

        self.x += displacement[0]
        self.y += displacement[1]


def Angle(p1 : CollisionParticle, p2 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here
    v1_x = p2._x - p1._x
    v1_y = p2._y - p1._y
    theta = np.arctan(v1_y/v1_x)
    print(f'Angle {theta}') # for debugging
    return theta

def makeparticles(x_coord_array):
    radius = (x[-1]-x[0])/(2*len(x))
    particles = []
    for n in x_coord_array:
        particles.append(CollisionParticle(n, np.float64(defined_function(n)), radius))
    return particles

# Was used to understang and debug collisions
def plotparticles(particles_array):
    # fig, ax = plt.subplots()
    for particle in particles_array:
        circle = plt.Circle((particle._x, particle._y), particle._r, color='r')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        # print((particle.x, particle.y))

# Collision detection function
def collision_detect(ball : Ball, particles_array):
    # Save all particles that collide with a ball 
    collision_particles = []
    for particle in particles_array:
        if particle._x <= ball._x + ball._r and particle._x >= ball._x - ball._r:

            vx = ball._x - particle._x
            vy = ball._y - particle._y
            distance = np.sqrt(np.square(vx) + np.square(vy))
            
            if distance < ball._r + particle._r:
                p = [particle, distance]
                collision_particles.append(p)

    if len(collision_particles) == 0:     
        return False, None, None
    
    # Iterate through the list to find the closest particle from all
    closest = collision_particles[0]
    for i in range(1, len(collision_particles)):
        if closest[1] > collision_particles[i][1]:
            closest = collision_particles[i]
    distance = closest[1]
    closest = closest[0]


    # Displacement correction. Maintains the ball on the graph, so it does not fall through it.
    vx = ball._x - closest._x
    vy = ball._y - closest._y
    v = np.sqrt(np.square(vx)+np.square(vy))
    displacement_value = ball._r + closest._r - distance
    theta = np.arccos(vx/v)
    print(f'Correction angle: {theta}')
    x_displacement = np.cos(theta) * displacement_value
    y_displacement = np.sin(theta) * displacement_value
    ball._x += x_displacement
    ball._y += y_displacement

    # Exception check if a particle is last on a graph, so an error does not show up.
    if not closest == particleArray[-1]:
        particle2 = particleArray[particleArray.index(closest) + 1]
    elif closest == particleArray[-1]:
        particle2 = closest
        closest = particleArray[particleArray.index(particle2) - 1]

    return True, closest, particle2

    # Function to get a velocity of a ball at any position
def get_velocity(g, H, h, miu, t):
    if H - h < 0 :
        return 0 
    velocity = np.sqrt((10 * g * (H-h)) / 7)
    # velocity = velocity - miu*velocity*t
    print(f'velocity {velocity}')
    return velocity

# Number of particles
resolution : int = 2000


# Function equation
formula : str = "0.1**x"
defined_function = string_to_function(formula)

# Starting values for the ball
ball = Ball( x = 0.5, y = 0.5, r=0.04)
ball._x = 0.25
ball._y = 0.65

# Calculating graph data
from_x = 0
to_x = 20
x = np.linspace(from_x, to_x, resolution)
y = defined_function(x)

# Making particles on the graph
particleArray = makeparticles(x)

# Starting values / constants
miu = 1
g = 9.81
H = ball._y
dt = 0.02
t = 0

x_displacement = 0
y_displacement = 0
sign = 1

# For animation, saves all the names of saved figure names, in order to retrieve them easier
filenames = []
max_frames = 300 #Duration of the animation

# Makes every frame/image of the graph with the ball
for i in range(0, max_frames):
    
    ax = plt.gca()
    ax.set_aspect( 1 ) # set aspect ratio

    ball.draw(ax=ax)

    # Variables from collision detect
    isColliding, CollidingWithP1, NextP = collision_detect(ball, np.array(particleArray))

    # For debugging
    print(f'H :{H}')
    print(f'h :{ball._y}')

    

    # Checks, whether there is a collision, and if there is, calculates x and y displacement
    if isColliding:

        H = H*(1-miu*dt*0.5) # Energy dissipation

        if velocity <= 0.001 and velocity >= -0.001: # Changes the sign (direction) of velocity, when it is 0 on the slope
            H = ball._y + 0.01
            sign = -sign

        # Calculations of displacements 
        x_displacement = np.cos(Angle(CollidingWithP1, NextP, ball)) * velocity * dt * sign
        y_displacement = np.sin(Angle(CollidingWithP1, NextP, ball)) * velocity * dt * sign

    # No collision - free fall
    else:
        ball.advance(dt=dt, accel=np.array([0,-10]))
        x_displacement = 0
        y_displacement = 0
    
    # Displaces the ball
    ball._x = ball._x + x_displacement
    ball._y = ball._y + y_displacement
    velocity = get_velocity(g, H, ball._y, miu, t)
    

    # For debugging
    print(f'ball._x {ball._x}')
    print(f'ball._y {ball._y}')
    print(f'x_displacement {x_displacement}')
    print(f'y_displacement {y_displacement}')
    print()
    
    # Draws a graph
    plt.title( f'Colored Circle Frame : {i}' )
    plt.plot(x,y)
    fname = f'Frames/frame{i}.png'
    filenames.append(fname) 
    plt.ylim(ymin=-0.5, ymax =1.5)
    plt.savefig(fname)
    plt.close()

# Makes an animation and saves as a gif
with imageio.get_writer('sim.gif', fps = 50) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print('gif saved')


# Deleting the folder with frames
if os.path.isdir('Frames'):
    shutil.rmtree('Frames')
