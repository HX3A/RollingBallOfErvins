# Bumba2, tikai pievienoju vēl "particles", 2 funkcijas : 1. izveido visus objektus, 2. uzzīmē garfiku no objektiem
# Vēl pievienoju divus veidus ātruma aprēķināšanai - ar enerģijas zudumiem un bez. Pievienoju gravitācijas funkciju. Un vēl Collision detection Lite, vajag viņu uztaisīt tā, lai nepārbauda visiem punktiem, bet daļai.
# Klassēs visur pieliku self._x (underscore), jo savādāk nestrādāja

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def string_to_function(expression):
    """Converts a string to a numpy function"""
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
    def __init__(self, x : float, y : float, r : float, vx: float = 0, vy:float = 0) -> None:
        self._x = x
        self._y = y
        self._r = r
        self._vx = vx
        self._vy = vy
        self._vel : np.ndarray = [vx, vy] 

    # for scalar -> Positive goues upwards
    # def advance(self, dt, accel : float):
    #     """Advance the Ball's position forward in time by dt."""
    #     displacement = ((accel*(dt**2)) / 2)
    #     # self._r += self.v * dt
    #     # self._x += displacement[0]
    #     self._y += displacement
    # for Vector -> Positive goues upwards

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, newx : float):
        self._x = newx

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, newy : float):
        self._y = newy

    @property
    def vel(self) -> np.ndarray:
        return np.array(self._vel)
    
    @vel.setter
    def vel(self, newVel : np.ndarray):
        self._vel = newVel

    def draw(self, ax):
        circle = plt.Circle((self._x, self._y), radius=self._r, color='b')
        ax.add_patch(circle)
        self.patch = circle
        return circle

    def Update(self):
        self.patch.center = [self._x, self._y]

    def advance(self, dt, accel : np.ndarray):
        """Advance the Ball's position forward in time by dt."""
        displacement = np.array(self._vel) * dt + ((accel*(dt**2)) / 2)

        # self._vx += velocity[0]
        # self._vy += velocity[1]

        self._x += displacement[0]
        self._y += displacement[1]

        self._vel += accel  * dt

    
    def advance2(self, dt,angle, velocity, accel : np.ndarray = (0,-10)):
        """Advance the Ball's position forward in time by dt."""
        # velocity = np.linalg.norm(velocity)
        # accel = np.linalg.norm(accel)

        # cos = np.cos(angle)
        # sin = np.sin(angle)

        # self._x += cos * velocity * dt + accel[0] * (dt ** 2) / 2
        # self._y += sin * velocity * dt + accel[1] * (dt ** 2) / 2
        # self._x += self._vel[0] * dt + accel[0] * (dt ** 2) / 2
        # self._y += self._vel[1] * dt + accel[1] * (dt ** 2) / 2

        self._x += self._vel[0] * dt
        self._y += self._vel[1] * dt

        self._vel = accel * dt


def TangentToParticle(p1 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here

    v1 : np.ndarray = np.array([ball._x, ball._y]) - np.array([p1._x, p1._y])
    v1x = v1[0]
    v1y = v1[1]

    #The direction may be flipped, just * by -1 if thats the case
    t1 : np.ndarray = np.array([-v1y, v1x])
    # print(t1, " - perpendicular vector") 
     
    return t1

def FlipVector90(v1):
    return np.array([-v1[1],v1[0]])


def Normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def Project(v1 , v2) -> np.ndarray:
    scalar_projection = np.dot(v2, v1)

    v_projection = Normalize(v2) * scalar_projection

    # print(Normalize(v2), "Project insides", v_projection, "v projection")

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
        circle = plt.Circle((particle._x, particle._y), particle._r, color='r')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        # print((particle._x, particle._y))


# def collision_detect(ball : Ball, particles_array):
#     for particle in particles_array:
#         distance = np.sqrt(np.square(ball._x - particle._x) + np.square(ball._y - particle._y))
#         # print(index,distance)
#         if distance < ball._r + particle._r:
#             print("True")
#             return True, particle
        
#     return False, None

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

    theta = np.arctan(vy/vx)
    
    x_displacement = np.cos(theta) * displacement_value
    y_displacement = np.sin(theta) * displacement_value

    ball._x += x_displacement
    ball._y += y_displacement

    particle2 = particleArray[particleArray.index(closest) + 1]

    return True, closest, particle2

def Angle(p1 : CollisionParticle, p2 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here

    v1_x = p2._x - p1._x
    v1_y = p2._y - p1._y
    theta = np.arctan(v1_y/v1_x)
    # print(f'Angle {theta}')
    # print(p1._x , p1._y)
    # print(p2._x , p2._y)
    return theta


resolution : int = 200

# function equation
# formula : str = "np.tanh(x)"
formula : str = "1*x**2-2*x+1.1"
# formula : str = "-0.1*x - 0.3"


defined_function = string_to_function(formula)

# x = np.array((1,2),0.01)
fromx = 0
tox = 2

x = np.linspace(fromx, tox, resolution)
y = defined_function(x)


ball = Ball( x = 0.5, y = 0.5, r=0.04)

particleArray = makeparticles(x)
# plotparticles(particleArray)
# print(type(particleArray))

dt = 0.01

ax = plt.gca()
fig = plt.gcf()
max_frames = 100

ball.draw(ax=ax)

def Move(i):
    ax.set_aspect( 1 ) # set aspect ratio

    ball.Update()

    isCollifing, CollidingWithP, NextP = collision_detect(ball, np.array(particleArray))

    if isCollifing: # nestrada, nezinu kapeec
        accelProjection = Project( np.array([0, -10]), TangentToParticle(CollidingWithP, ball))
        accelNormal = FlipVector90(accelProjection)

        # FullAccel = accelProjection + accelNormal
        # FullAccel = np.array([0, -10]) + accelNormal
        FullAccel = accelProjection + accelNormal
        # ball._vel += np.array(accelProjection) * dt
        if FullAccel[0] < 0:
            FullAccel[1] *= -1
        # ball.advance(dt,velocity= ball.vel, accel=np.array(accelProjection))
        
        theta = Angle(CollidingWithP, NextP, ball)
        
        # if

        ball.advance2(dt, theta, ball._vel, accelProjection * 100)

        # print(FullAccel)
        print(ball._vel)


        ax.quiver(ball.x,ball.y,  *ball._vel ,scale = .0001, color = "red")
        # print(ball._vel)
        
    else:
        ball.advance(dt, accel=np.array([0,-10]))
    
    plt.title( f'Colored Circle Frame : {i}' )
    plt.plot(x,y)


anim = animation.FuncAnimation(fig,Move, frames=max_frames,interval=100 )

plt.show()
