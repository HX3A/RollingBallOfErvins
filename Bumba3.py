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
    # pass all the 
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

    v1 : np.ndarray = np.array([ball._x, ball._y]) - np.array([p1._x, p1._y])
    v1_x = v1[0]
    v1_y = v1[1]

    #The direction may be flipped, just * by -1 if thats the case
    t1 : np.ndarray = np.array([v1_x, -v1_y])
    theta = np.arctan(v1_x/-v1_y) 
    print(f'Angle {theta}')
    return theta
    

def Angle(p1 : CollisionParticle, p2 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here

    v1_x = p2._x - p1._x
    v1_y = p2._y - p1._y
    theta = np.arctan(v1_y/v1_x)
    print(f'Angle {theta}')
    # print(p1._x , p1._y)
    # print(p2._x , p2._y)
    return theta

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
    # fig, ax = plt.subplots()
    for particle in particles_array:
        circle = plt.Circle((particle._x, particle._y), particle._r, color='r')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        # print((particle.x, particle.y))


def collision_detect(ball : Ball, particles_array):
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
    
    closest = collision_particles[0]
    for i in range(1, len(collision_particles)):
        if closest[1] > collision_particles[i][1]:
            closest = collision_particles[i]
    distance = closest[1]
    closest = closest[0]

    vx = ball._x - closest._x
    vy = ball._y - closest._y
    displacement_value = ball._r + closest._r - distance
    theta = np.arccos(vx/distance)
    print(f'Correction angle: {theta}')
    x_displacement = np.cos(theta) * displacement_value
    y_displacement = np.sin(theta) * displacement_value
    ball._x += x_displacement
    ball._y += y_displacement

    if not closest == particleArray[-1]:
        particle2 = particleArray[particleArray.index(closest) + 1]
    elif closest == particleArray[-1]:
        particle2 = closest
        closest = particleArray[particleArray.index(particle2) - 1]

    return True, closest, particle2


def get_velocity(g, H, h):
    velocity = np.sqrt((10 * g * (H-h) ) / 7)
    print(f'velocity {velocity}')
    return velocity

def get_velocity_with_miu(g, F, m, H, h): 
    velocity = np.sqrt(((10 * g * (H-h)) + F) / (7 * m))
    return velocity


resolution : int = 2000


# function equation
formula : str = "x**2 -2*x +1"
defined_function = string_to_function(formula)


ball = Ball( x = 0.5, y = 0.5, r=0.04)
ball._x = 0.25
ball._y = 0.65


from_x = 0
to_x = 3


x = np.linspace(from_x, to_x, resolution)
y = defined_function(x)


particleArray = makeparticles(x)

m = 1
F = 0.5
g = 9.81
H = ball._y
dt = 0.02

x_displacement = 0
y_displacement = 0

filenames = []
max_frames = 200
for i in range(0, max_frames):

    ax = plt.gca()
    ax.set_aspect( 1 ) # set aspect ratio

    ball.draw(ax=ax)

    isCollifing, CollidingWithP1, NextP  = collision_detect(ball, np.array(particleArray))

    ball._x = ball._x + x_displacement
    ball._y = ball._y + y_displacement

    if isCollifing: 
        velocity = get_velocity(g, H, ball._y)
        x_displacement = np.cos(Angle(CollidingWithP1, NextP, ball))*velocity * dt * 0.5
        y_displacement = np.sin(Angle(CollidingWithP1, NextP, ball))*velocity * dt * 0.5
    else:
        ball.advance(dt=dt, accel=np.array([0,-10]))
        x_displacement = 0
        y_displacement = 0

    # print(f'ball._x {ball._x}')
    # print(f'ball._y {ball._y}')
    # print(f'x_displacement {x_displacement}')
    # print(f'y_displacement {y_displacement}')
    
    plt.title( f'Colored Circle Frame : {i}' )
    # plotparticles(particleArray)
    plt.plot(x,y)
    fname = f'Frames/frame{i}.png'
    filenames.append(fname) 
    plt.ylim(ymin=-0.5, ymax =1.5)
    plt.savefig(fname)
    # plt.show()
    plt.close()

with imageio.get_writer('sim.gif', fps = 20) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print('gif saved')


# Deleting the folder with frames
if os.path.isdir('Frames'):
    shutil.rmtree('Frames')


# Deleting the folder with frames
if os.path.isdir('Frames'):
    shutil.rmtree('Frames')
