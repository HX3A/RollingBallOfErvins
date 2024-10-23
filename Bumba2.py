import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
# from itertools import combinations

class Particle:
    """A class representing a two-dimensional ball"""

    def __init__(self, x = 1, y = 1, vx = 0, vy = 0, radius=0.01, mass = 1 ,  styles=None):
        """Initialize the particle's position, velocity, and radius."""

        self.r = np.array((x, y))
        self.v = np.array((vx, vy)) # velocity vector
        self.radius = radius

        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'edgecolor': 'b', 'fill': False}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.
    @property
    def x(self):
        return self.r[0]
    @x.setter
    def x(self, value):
        self.r[0] = value
    @property
    def y(self):
        return self.r[1]
    @y.setter
    def y(self, value):
        self.r[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        """Does the circle of this ball overlap?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.r += self.v * dt

        # Make the Particles bounce off the walls
        # if self.x - self.radius < 0:
        #     self.x = self.radius
        #     self.vx = -self.vx
        # if self.x + self.radius > 1:
        #     self.x = 1-self.radius
        #     self.vx = -self.vx
        # if self.y - self.radius < 0:
        #     self.y = self.radius
        #     self.vy = -self.vy
        # if self.y + self.radius > 1:
        #     self.y = 1-self.radius
        #     self.vy = -self.vy

class  Constants:
    g = 9.81
    nu = 0.5

    def __init__(self, nu = 0.5, g = 9.81) -> None:
        self.nu = nu
        self.g = g


        pass


def string_to_function(expression):
    def function(x):
        return eval(expression)
    return np.frompyfunc(function, 1, 1)


if __name__ == '__main__':
    # x0 = 0
    # y0 = 0

    # vx0 = 0
    # vy0 = 0

    resolution : int = 100

    # need to get a function equation
    formula : str = "1*x**2-2*x+1"
    # formula : str = "np.tanh(x)"
    defined_function = string_to_function(formula)

    # x = np.array((1,2),0.01)
    x = np.linspace(0, 1, resolution)
    y = defined_function(x)

    plt.plot(x,y)
    plt.show()


    # print("Finished main!")

#     nparticles = 50
#     radii = np.random.random(nparticles)*0.03+0.02
#     styles = {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}
#     sim = Simulation(nparticles, radii, styles)
#     sim.do_animation(save=False)