import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Simplex Algorithm Visualizer", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: bold; color: #89b4fa; }
    .card { background-color: #1e1e2e; padding: 15px; border-radius: 10px; border: 1px solid #45475a; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Simplex Algorithm Visualizer</div>", unsafe_allow_html=True)
st.caption("Interactive 2D/3D Geometry, Student Tableau Validation, Ratio Calculations, and Orthographic Projections.")

# --- SIDEBAR: PROBLEM CONFIGURATION ---
st.sidebar.header("1. Problem Parameters")

num_constraints = st.sidebar.number_input("Number of Constraints", min_value=1, max_value=6, value=2, step=1)

st.sidebar.subheader("Objective Function (Max Z)")
col_c1, col_c2 = st.sidebar.columns(2)
c1 = col_c1.number_input("c1 (for x1)", value=2.0, step=0.5)
c2 = col_c2.number_input("c2 (for x2)", value=1.0, step=0.5)

st.sidebar.subheader("Constraints")
A_inputs = []
b_inputs = []
ops = []

# Default values for initial load
default_A = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
default_b = [2.0, 1.0, 5.0, 5.0, 5.0, 5.0]

for i in range(int(num_constraints)):
    st.sidebar.markdown(f"**Constraint {i+1}**")
    ca1, ca2, cop, crhs = st.sidebar.columns([2, 2, 2, 2])
    
    def_a1 = default_A[i][0] if i < len(default_A) else 1.0
    def_a2 = default_A[i][1] if i < len(default_A) else 1.0
    def_rhs = default_b[i] if i < len(default_b) else 5.0
    
    a1_val = ca1.number_input(f"x1 (C{i+1})", value=def_a1, key=f"a1_{i}")
    a2_val = ca2.number_input(f"x2 (C{i+1})", value=def_a2, key=f"a2_{i}")
    op_val = cop.selectbox(f"Op (C{i+1})", ["<=", ">="], index=0, key=f"op_{i}")
    rhs_val = crhs.number_input(f"RHS (C{i+1})", value=def_rhs, key=f"rhs_{i}")
    
    mult = -1.0 if op_val == ">=" else 1.0
    A_inputs.append([a1_val * mult, a2_val * mult])
    b_inputs.append(rhs_val * mult)
    ops.append(op_val)

A = np.array(A_inputs)
b_vec = np.array(b_inputs)
n_s = len(b_vec)
headers = ['x1', 'x2'] + [f's{i+1}' for i in range(n_s)]

# --- SIMPLEX SOLVER ENGINE ---
def compute_simplex_iterations(c1, c2, A, b_vec):
    n_s = len(b_vec)
    c_obj = np.array([-c1, -c2] + [0.0] * n_s)
    
    table = np.zeros((n_s + 1, 2 + n_s + 1))
    table[:-1, :2] = A
    table[:-1, 2:2+n_s] = np.eye(n_s)
    table[:-1, -1] = b_vec
    table[-1, :-1] = c_obj
    
    basis = [2 + i for i in range(n_s)]
    iterations = []
    
    for _ in range(10):
        z_row = table[-1, :-1]
        
        curr_pt = [0.0, 0.0]
        for idx, b_var in enumerate(basis):
            if b_var < 2:
                curr_pt[b_var] = table[idx, -1]
                
        if np.all(z_row >= -1e-5):
            iterations.append({
                'table': table.copy(), 'basis': list(basis), 'z_row': z_row.copy(),
                'key_col': -1, 'key_row': -1, 'ratios': [], 'is_optimal': True, 'pt': curr_pt
            })
            break
            
        key_col = int(np.argmin(z_row))
        
        col_vals = table[:-1, key_col]
        rhs_vals = table[:-1, -1]
        ratios = []
        for cv, rv in zip(col_vals, rhs_vals):
            if cv > 1e-5:
                ratios.append(rv / cv)
            else:
                ratios.append(np.inf)
                
        key_row = int(np.argmin(ratios)) if any(r < np.inf for r in ratios) else -1
        
        iterations.append({
            'table': table.copy(), 'basis': list(basis), 'z_row': z_row.copy(),
            'key_col': key_col, 'key_row': key_row, 'ratios': ratios, 'is_optimal': False, 'pt': curr_pt
        })
        
        if key_row == -1:
            break
            
        pivot = table[key_row, key_col]
        table[key_row, :] /= pivot
        for r in range(n_s + 1):
            if r != key_row:
                table[r, :] -= table[r, key_col] * table[key_row, :]
                
        basis[key_row] = key_col
        
    return iterations

iterations = compute_simplex_iterations(c1, c2, A, b_vec)

# --- DISPLAY STANDARD FORM ---
st.markdown("### Standard Form Setup")
sf_html = f"<b>Maximize z - {c1}x<sub>1</sub> - {c2}x<sub>2</sub>" + "".join([f" - 0s<sub>{i+1}</sub>" for i in range(n_s)]) + " = 0</b><br>"
sf_html += "Subject to constraints:<br>"
for i in range(n_s):
    slacks = "".join([f" + 1s<sub>{j+1}</sub>" if i==j else f" + 0s<sub>{j+1}</sub>" for j in range(n_s)])
    sf_html += f"&nbsp;&nbsp;{A[i][0]}x<sub>1</sub> + {A[i][1]}x<sub>2</sub>{slacks} = {b_vec[i]}<br>"
sf_html += f"&nbsp;&nbsp;x<sub>1</sub>, x<sub>2</sub>, s<sub>1</sub>..s<sub>{n_s}</sub> &ge; 0"

st.markdown(f"<div class='card'>{sf_html}</div>", unsafe_allow_html=True)

# --- ITERATION NAVIGATION ---
max_step = len(iterations) - 1
selected_step = st.slider("Select Simplex Iteration Step", min_value=0, max_value=max_step, value=0, step=1)

iter_data = iterations[selected_step]

# --- STUDENT TABLEAU EVALUATION SECTION ---
st.markdown(f"### Iteration {selected_step} Tableau Evaluation")

if iter_data['is_optimal']:
    pt = iter_data['pt']
    max_z = iter_data['table'][-1, -1]
    st.success(f"**OPTIMAL SOLUTION REACHED**: x1 = {pt[0]:.2f}, x2 = {pt[1]:.2f} | Max Z = {max_z:.2f}")
else:
    st.info("Fill up the z-row entries to evaluate optimality:")
    
    input_cols = st.columns(len(headers))
    user_z_inputs = []
    for idx, h in enumerate(headers):
        val = input_cols[idx].number_input(f"{h}", value=0.0, step=0.5, key=f"z_in_{selected_step}_{idx}")
        user_z_inputs.append(val)
        
    if st.button("Verify z-row Values"):
        expected_z = iter_data['z_row']
        if np.allclose(user_z_inputs, expected_z, atol=1e-1):
            st.success("Verification Passed! Solutions match tableau expectations.")
        else:
            st.error("Incorrect entries detected!")
            for idx, (uv, ev) in enumerate(zip(user_z_inputs, expected_z)):
                if not np.isclose(uv, ev, atol=1e-1):
                    st.write(f"• Variable **{headers[idx]}**: Expected `{ev:.2f}`, got `{uv:.2f}`")

    # Display Pivot calculations
    k_col = iter_data['key_col']
    k_row = iter_data['key_row']
    entering_var = headers[k_col]
    leaving_var = headers[iter_data['basis'][k_row]]
    
    st.markdown("**Iteration Decision Variables & Replacement Ratios:**")
    st.write(f"• **Key Column (Entering):** `{entering_var}` (Most negative z-row value)")
    st.write(f"• **Key Row (Leaving):** `{leaving_var}` (Minimum positive replacement ratio)")
    
    ratio_str = ""
    for i, r in enumerate(iter_data['ratios']):
        b_var_name = headers[iter_data['basis'][i]]
        ratio_str += f"Row {i+1} ({b_var_name}): `{r:.2f}` | " if r < np.inf else f"Row {i+1} ({b_var_name}): `∞` | "
    st.caption(ratio_str)

# --- GRAPHICAL VISUALIZATIONS ---
st.markdown("---")
st.markdown("### Geometric Views & Engineering Projections")

tab1, tab2 = st.tabs(["3D Objective Surface", "2D Corner Point & Projections"])

curr_pt = iter_data['pt']

with tab1:
    x1_a = np.linspace(0, 4, 30)
    x2_a = np.linspace(0, 4, 30)
    X1, X2 = np.meshgrid(x1_a, x2_a)
    Z = c1 * X1 + c2 * X2
    
    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='Viridis', opacity=0.6, showscale=False))
    
    path_pts = [it['pt'] for it in iterations[:selected_step+1]]
    px = [p[0] for p in path_pts]
    py = [p[1] for p in path_pts]
    pz = [c1*x + c2*y for x, y in zip(px, py)]
    
    fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers+text',
                                marker=dict(size=8, color='#f38ba8'),
                                line=dict(color='#f38ba8', width=5),
                                text=[f"Iter {i}" for i in range(len(px))]))
    
    fig3d.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='Z Elevation'),
                        margin=dict(l=0, r=0, b=0, t=30), height=450, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)

with tab2:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 2D Trajectory
    ax1 = axes[0]
    x_grid = np.linspace(0, 4, 300)
    for i in range(len(b_vec)):
        a1_val, a2_val = A[i]
        if a2_val != 0:
            y_grid = (b_vec[i] - a1_val*x_grid) / a2_val
            ax1.plot(x_grid, y_grid, label=f'C{i+1}: {a1_val}x1+{a2_val}x2<={b_vec[i]}', linestyle='--')
        else:
            ax1.axvline(x=b_vec[i]/a1_val, label=f'C{i+1}: {a1_val}x1<={b_vec[i]}', linestyle='--')
            
    if not iter_data['is_optimal'] and iter_data['key_row'] != -1:
        next_pt = iterations[selected_step + 1]['pt'] if selected_step + 1 < len(iterations) else curr_pt
        if next_pt != curr_pt:
            ax1.annotate('', xy=(next_pt[0], next_pt[1]), xytext=(curr_pt[0], curr_pt[1]),
                         arrowprops=dict(facecolor='#fab387', edgecolor='#fab387', shrink=0.08, width=2, headwidth=8))

    ax1.scatter([curr_pt[0]], [curr_pt[1]], color='#f38ba8', s=120, zorder=5, label='Current Point')
    ax1.set_xlim(-0.2, 3.5)
    ax1.set_ylim(-0.2, 3.5)
    ax1.set_title("2D Simplex Corner Path", fontweight='bold', color='#89b4fa')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='upper right')

    # Top Projection
    ax2 = axes[1]
    ax2.scatter([curr_pt[0]], [curr_pt[1]], color='#89dceb', s=100)
    ax2.set_title("Top View (x1 vs x2)", fontweight='bold', color='#89dceb')
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_xlim(-0.2, 3.5)
    ax2.set_ylim(-0.2, 3.5)
    ax2.grid(True, alpha=0.3)

    # Side Projection
    ax3 = axes[2]
    curr_z = c1*curr_pt[0] + c2*curr_pt[1]
    ax3.plot(px, pz, 'm--o', label='Trajectory Path')
    ax3.scatter([curr_pt[0]], [curr_z], color='#a6e3a1', s=120, label='Current Z')
    ax3.set_title("Side View (x1 vs Z Elevation)", fontweight='bold', color='#a6e3a1')
    ax3.set_xlabel("x1")
    ax3.set_ylabel("Z Value")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
