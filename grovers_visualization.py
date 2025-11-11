import numpy as np
import matplotlib.pyplot as plt

N = 16  
theta = np.arcsin(1/np.sqrt(N))  
v_s = np.array([np.cos(theta), np.sin(theta)]) 
v_oracle = np.array([v_s[0], -v_s[1]])        

def reflect(u, v):
    return 2 * (np.dot(u, v) / np.dot(u, u)) * u - v

v_after_diffusion = reflect(v_s, v_oracle)  

def grover_iteration(vec, oracle_axis, s_axis):
    v = np.array(vec)
    v = np.array([v[0], -v[1]]) 
    v = reflect(s_axis, v)      
    return v

vectors = [v_s]
v = v_s.copy()
for i in range(4):
    v = grover_iteration(v, None, v_s)
    vectors.append(v)

fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

circle = plt.Circle((0,0), 1.0, fill=False, linewidth=0.7)
ax.add_artist(circle)
ax.axhline(0, linewidth=0.6)
ax.axvline(0, linewidth=0.6)

ax.text(1.02, -0.03, 'rest (|r⟩) — x axis', va='center')
ax.text(-0.03, 1.02, 'answer (|w⟩) — y axis', ha='right')

ax.arrow(0, 0, v_s[0], v_s[1], head_width=0.03, length_includes_head=True)
ax.text(v_s[0]*1.05, v_s[1]*1.05, '|s⟩ initial', va='bottom', ha='left')

ax.arrow(0, 0, v_oracle[0], v_oracle[1], head_width=0.03, length_includes_head=True)
ax.text(v_oracle[0]*1.05, v_oracle[1]*1.05, 'after oracle (flip |w⟩)', va='top', ha='left')

ax.arrow(0, 0, v_after_diffusion[0], v_after_diffusion[1], head_width=0.03, length_includes_head=True)
ax.text(v_after_diffusion[0]*1.05, v_after_diffusion[1]*1.05, 'after diffusion', va='bottom', ha='right')

for i, vec in enumerate(vectors[1:], start=1):
    ax.plot([0, vec[0]], [0, vec[1]], linewidth=1, linestyle='--')
    ax.text(vec[0]*0.85, vec[1]*0.85, f'iter {i}', fontsize=9, alpha=0.8)

angle_start = 0
angle_theta = theta
angle_2theta = 2*theta
arc_t = np.linspace(angle_start, angle_start + angle_2theta, 60)
arc_x = 0.18 * np.cos(arc_t)
arc_y = 0.18 * np.sin(arc_t)
ax.plot(arc_x, arc_y, linewidth=1)
ax.text(0.18*np.cos(angle_start + 0.5*angle_2theta),
        0.18*np.sin(angle_start + 0.5*angle_2theta),
        'rotation 2θ', fontsize=9, ha='center', va='center')

ax.set_title("Geometric visualization of Grover's algorithm\ny axis = answer |w⟩, x axis = everything else |r⟩")
ax.set_xlabel('Amplitude on rest states (x)')
ax.set_ylabel('Amplitude on answer state (y)')

plt.show()
