import numpy as np
from numpy import pi, linspace, array, dot, sin, cos, diag, concatenate, zeros
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def D(q, params):
    alpha_1, alpha_2 = q
    l, m, J, b, g = params
    
    d11 = (m[0]+m[1]) * l[0]**2  + J[0]
    d12 = m[1]* l[0]* l[1]* cos(alpha_1 - alpha_2)
    d21 = d12
    d22 = m[1] * l[1]**2 + J[1]
    return array([[d11,d12],[d21,d22]])

def c_term(q, dq, params):
    alpha_1, alpha_2 = q
    dalpha_1, dalpha_2 = dq

    c1 = m[1]* l[0]* l[1]* sin(alpha_1 - alpha_2)* dalpha_2**2
    c2 = -m[1]* l[0]* l[1]* sin(alpha_1 - alpha_2)* dalpha_1**2
    return array([c1, c2])

def g_term(q, params):
    alpha_1, alpha_2 = q

    g1 = (m[0]+m[1])*g*l[0]*cos(alpha_1)
    g2 = m[1]*g*l[1]*cos(alpha_2)
    return array([g1, g2])

def Q_d(q, dq, params):
    
    dalpha_1, dalpha_2 = dq
    Q_d_1 = b[0]*dalpha_1
    Q_d_2 = b[1]*dalpha_2
    return array([Q_d_1, Q_d_2])

def h(q, dq, params):
    return c_term(q, dq, params) + g_term(q, params)


def sysode(x, t, control, params, control_params):
    q, dq = x[:2], x[2:4]

    D_c = D(q, params)
    h_c = h(q, dq, params)
    Q_d_c = Q_d(q, dq, params)

    # Calculate control
    u = control(x, t, control_params)
    ddq = dot(inv(D_c), u - Q_d_c -  h_c )

    dx1 = dq
    dx2 = ddq
    dx = dx1, dx2

    return concatenate(dx)

# Manipulator parameters
l = 0.3, 0.3
m = 0.5 , 3.0
J = 0.01, 0.01
b = 0.0, 0.0
g = 9.81
params = l, m, J, b, g

def control(x, t, control_params):
    q, dq = x[:2], x[2:4]
    gains = control_params['gains']
    K1, K2 = gains
    
    q_d = pi/2, pi/4
    q_e = q_d - q

    #u =  dot(K1, q_e) + dot(K2, - dq)
    u=0
    return u
    
control_params = {}
Kp = diag([150, 150])
Kd = diag([15, 15])
control_params['gains'] = Kp, Kd
control_params['q_d'] = [pi/4, pi/3]

from scipy.integrate import odeint


# Integration
t0 = 0 # Initial time 
tf = 10 # Final time
N = 2E3 # Numbers of points in time span
t = linspace(t0, tf, int(N)) # Create time span
x0 = [1.2*np.pi/4, 0,1.2*np.pi/4, 0] # Set initial state 
sol = odeint(sysode, x0, t,
             
             args=(control, params, control_params,)) # Integrate system
q, dq = sol[:,:2], sol[:,2:4]
alpha_1, alpha_2 = q[:,0], q[:,1]


from matplotlib.pyplot import *
plot(t, alpha_1,'r', linewidth=2.0, label = 'Joint 1')
plot(t, alpha_2,'b', linewidth=2.0, label = 'Joint 2')
# plot(t, cos(2*pi*t),'black', linestyle = '--', alpha = 0.5,  linewidth=2.0, label = 'Joint 1 des')
# plot(t, sin(2*pi*t),'black', linestyle = '--', alpha = 0.5, linewidth=2.0, label = 'Joint 2 des')

# hlines(cos(2*pi*t), sin(2*pi*t), t0, tf,color = 'black', linestyle = '--', alpha = 0.7)
hlines(0, t0, tf,color = 'black', linestyle = '--', alpha = 0.7)
hlines(pi/2, t0, tf,color = 'black', linestyle = '--', alpha = 0.7)
# plot(t, alpha_1_exct,'r--', linewidth=2.0, alpha = 0.6)
# plot(t, alpha_2_exct,'b--', linewidth=2.0, alpha = 0.6)
grid(color='black', linestyle='--', linewidth=1.0, alpha = 0.7)
grid(True)
xlim([0, tf])
legend()
ylabel(r'Angles $q$ (rad)')
xlabel(r'Time $t$ (s)')
show()

np.sin(alpha_1)
#for colab
!mkdir frames

!mkdir gif

# Convert to Cartesian coordinates of the two bob positions.
y1 = l[0] * np.sin(alpha_1)
x1 = -l[1] * np.cos(alpha_1)
y2 = y1 + l[0] * np.sin(alpha_2)
x2 = x1 - l[1] * np.cos(alpha_2)

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 19, 0.01
t = np.arange(0, tmax+dt, dt)

r = 0.02
# Plot a trail of the m2 bob's position for the last trail_secs seconds.
trail_secs = 1
# This corresponds to max_trail time points.
max_trail = int(trail_secs / dt)

def make_plot(i):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='g', ec='g', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='b', ec='b', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='b', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-l[0]-l[0]-r, l[0]+l[0]+r)
    ax.set_ylim(-l[0]-l[0]-r, l[0]+l[0]+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
    plt.cla()


# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
fps = 10
di = int(1/fps/dt)
fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)

for i in range(0, t.size, di):
    #print(i // di, '/', t.size // di)
    make_plot(i)

import glob
from PIL import Image, ImageSequence

# filepaths
fp_in = "/content/frames/*.png"
fp_out = "/content/gif/Simulation_no_f.gif"

imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=1)