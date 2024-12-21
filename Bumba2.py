import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from Functions import *

resolution : int = 400

# function equation
# formula : str = "np.tanh(x)"
formula : str = "0.78*x**2-1.5*x+0.8"
# formula : str = "-0.1*x - 0.3"

defined_function = string_to_function(formula)

fromx = 0
tox = 2

x = np.linspace(fromx, tox, resolution)
y = defined_function(x)

ball = Ball( x = 0.5, y = 0.45, r=0.04)

particleArray = makeparticles(x, defined_function,x)
# plotparticles(particleArray)
# print(type(particleArray))

dt = 0.01

ax = plt.gca()
fig = plt.gcf()

ax.set_aspect( 1 ) # set aspect ratio to 1

max_frames = 100

ball.draw(ax=ax)

# define gravity
g = np.array([0,-10])

def Move(i):
    # Draw new ball, replacing the old iteration
    ball.Update()

    isColliding, CollidingWithP, NextP = collision_detect(ball, np.array(particleArray), particleArray)

    if isColliding: 
        # Direction, in which the particle will move
        VelProjection =  Normalize(Project( ball._vel, TangentToParticle(CollidingWithP, ball)))
        AccelProjection =  Project( g, VelProjection) #* np.linalg.norm(g)
        # ball._vel = VelProjection 

        # Corrections based on distance from the closest particle
        correction : float = CollisionCorrection(CollidingWithP, ball)

        # depends on the distance between ball and collision points * (r_ball + r_p) / d
        NormAccel = - FlipVector90(AccelProjection) * np.linalg.norm(g) * correction

        # update the ball 
        ball.advance2(dt,AccelProjection + NormAccel )

        # ax.quiver(ball.x,ball.y, *VelProjection ,scale = .0001, color = "red")
    else:
        ball.advance2(dt,g )


anim = animation.FuncAnimation(fig,Move, frames=500,interval=1 )
plt.plot(x,y)
plt.show()
# anim.save("save")

writer = animation.PillowWriter(fps=24,
                                metadata=dict(artist='E.Laizans un  E.Karavackis'),
                                bitrate=1800)

anim.save('scatter.gif', writer=writer)