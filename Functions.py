import numpy as np
from matplotlib import pyplot as plt

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
    # pass all the settings
    def __init__(self, x : float, y : float, r : float, vx: float = 0, vy:float = 0) -> None:
        self._x = x
        self._y = y
        self._r = r
        self._vx = vx
        self._vy = vy
        self._vel : np.ndarray = [vx, vy] 

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

    def advance(self, dt, accel: np.ndarray):
        """Advance the Ball's position forward in time by dt."""
        # Update velocity first
        # Calculate displacement using updated velocity
        self._vel += accel * dt  # Update velocity based on acceleration
        displacement = self._vel * dt + (accel * (dt ** 2)) / 2

        self._x += displacement[0]
        self._y += displacement[1]

    
    def advance2(self, dt, accel : np.ndarray = (0,-10)):
        """Advance the Ball's position forward in time by dt while colliding."""
        self._vel += accel * dt

        self._x += self._vel[0] * dt + (accel[0] * dt**2) / 2 
        self._y += self._vel[1] * dt + (accel[1] * dt**2) / 2



def TangentToParticle(p1 : CollisionParticle, ball : Ball) -> np.ndarray:
    # pass the CLOSEST particle in here
    # returns vector, tangent to the vector connecting p1 to ball

    v1 : np.ndarray = np.array([ball._x, ball._y]) - np.array([p1._x, p1._y])
    v1x = v1[0]
    v1y = v1[1]

    # perpendicular vector
    t1 : np.ndarray = np.array([-v1y, v1x])
     
    return t1

def FlipVector90(v1):
    return np.array([-v1[1],v1[0]])


def Normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def Project(v1 , v2) -> np.ndarray:
    scalar_projection = - np.abs( np.dot(v2, v1))

    v_projection = Normalize(v2) * scalar_projection

    return v_projection 

def Project2(x, y) -> float:
    return np.dot(x, y) / np.linalg.norm(y)

def makeparticles(x_coord_array, defined_function, x):
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

def collision_detect(ball : Ball, particles_array, particleArray):
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

    # vx = ball._x - closest._x
    # vy = ball._y - closest._y
    # displacement_value = ball._r + closest._r - distance

    # theta = np.arctan(vy/vx)
    
    # x_displacement = np.cos(theta) * displacement_value
    # y_displacement = np.sin(theta) * displacement_value

    # ball._x += x_displacement
    # ball._y += y_displacement

    particle2 = particleArray[particleArray.index(closest) + 1]

    return True, closest , particle2

def Angle(p1 : CollisionParticle, p2 : CollisionParticle ):
    v1_x = p2._x - p1._x
    v1_y = p2._y - p1._y
    theta = np.arctan(v1_y/v1_x)
    return theta

def CollisionCorrection(CollidingWithP : CollisionParticle, ball : Ball) -> float:
        d = np.sqrt(np.square(ball._x - CollidingWithP._x) + np.square(ball._y - CollidingWithP._y))
        r_sum = ball._r + CollidingWithP._r

        # danger!!! div by 0
        correction : float = r_sum / d

        if correction >= 400 :
            return 400
        else :
            return correction