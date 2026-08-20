import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Simplex Method Practice & Visualizer", layout="wide")

# Custom CSS for UI styling, blinking error effects, and table layouts
st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: bold; color: #89b4fa; }
    .card { background-color: #1e1e2e; padding: 15px; border-radius: 10px; border: 1px solid #45475a; margin-bottom: 15px; }
    
    /* Blinking error styling for incorrect inputs */
    @keyframes blinker {
        50% { opacity: 0.2; background-color: #f85149; }
    }
    .blink-error {
        border: 2px solid #f85149 !important;
        border-radius: 5px;
        animation: blinker 1s linear infinite;
        padding: 5px;
        color: #ff7b72;
        font-weight: bold;
    }
    .pivot-highlight {
        background-color: #238636;
        color: white;
        padding: 8px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Interactive Simplex Tableau & Step Visualizer</div>", unsafe_allow_html=True)
st.caption("Practice Row-Reduction Calculations | Automated Error Validation | Interactive 2D/3D Geometry")

# --- SIDEBAR: PROBLEM INPUTS ---
st.sidebar.header("1. Problem Formulation")

num_constraints = st.sidebar.number_input("Number of Constraints", min_value=1, max_value=4, value=2, step=1)

st.sidebar.subheader("Objective Function (Max Z)")
col_c1, col_c2 = st.sidebar.columns(2)
c1 = col_c1.number_input("c1 (for x1)", value=2.0, step=0.5)
c2 = col_c2.number_input("c2 (for x2)", value=1.0, step=0.5)

st.sidebar.subheader("Constraints")
A_inputs, b_inputs, ops = [], [], []

default_A = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
default_b = [2.0, 1.0, 5.0, 5.0]

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
cj_coeffs = [c1, c2] + [0.0] * n_s

# --- SIMPLEX ALGORITHM ENGINE ---
def compute_all_simplex_iterations(c1, c2, A, b_vec):
    n_s = len(b_vec)
    c_obj = np.array([-c1, -c2] + [0.0] * n_s)
    
    table = np.zeros((n_s + 1, 2 + n_s + 1))
    table[:-1, :2] = A
    table[:-1, 2:2+n_s] = np.eye(n_s)
    table[:-1, -1] = b_vec
    table[-1, :-1] = c_obj
    
    basis = [2 + i for i in range(n_s)] # Initial slack basis indices
    iterations = []
    
    for _ in range(10):
        z_row = table[-1, :-1]
        
        curr_pt = [0.0, 0.0]
        for idx, b_var in enumerate(basis):
            if b_var < 2:
                curr_pt[b_var] = table[idx, -1]
                
        # Check Optimality: All z_row coefficients >= 0 (for standard max form z - c_j)
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

iterations = compute_all_simplex_iterations(c1, c2, A, b_vec)

# --- SESSION STATE INITIALIZATION ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'step_verified' not in st.session_state:
    st.session_state.step_verified = False

step = st.session_state.current_step
iter_data = iterations[min(step, len(iterations)-1)]

# --- DISPLAY STANDARD FORM FORMULATION ---
st.markdown("### Standard Form Notation")
sf_html = f"<b>Maximize Z = {c1}x<sub>1</sub> + {c2}x<sub>2</sub>" + "".join([f" + 0s<sub>{i+1}</sub>" for i in range(n_s)]) + "</b><br>"
sf_html += "Subject to:<br>"
for i in range(n_s):
    slacks = "".join([f" + 1s<sub>{j+1}</sub>" if i==j else f" + 0s<sub>{j+1}</sub>" for j in range(n_s)])
    sf_html += f"&nbsp;&nbsp;{A[i][0]}x<sub>1</sub> + {A[i][1]}x<sub>2</sub>{slacks} = {b_vec[i]}<br>"
sf_html += f"&nbsp;&nbsp;x<sub>1</sub>, x<sub>2</sub>, s<sub>1</sub>..s<sub>{n_s}</sub> &ge; 0"
st.markdown(f"<div class='card'>{sf_html}</div>", unsafe_allow_html=True)

# --- SIMPLEX TABLE PRACTICE SECTION ---
st.markdown(f"### Iteration {step} — Simplex Calculation Practice Table")
st.write("Fill up the **core body values**, **solution vector (b)**, and **Z-row (Cj - Zj / Row Z)** for this iteration:")

# Table Headers Display
header_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.5, 1.5])
header_cols[0].markdown("**CB**")
header_cols[1].markdown("**Basis**")
for idx, h in enumerate(headers):
    header_cols[2 + idx].markdown(f"**{h}** (Cj={cj_coeffs[idx]})")
header_cols[-2].markdown("**b (X_B)**")
header_cols[-1].markdown("**Ratio**")

# Prepare student input containers
user_table_inputs = []
user_z_inputs = []

expected_table = iter_data['table']
basis_indices = iter_data['basis']

# Render rows for basic variables
for i in range(n_s):
    b_var_idx = basis_indices[i]
    b_var_name = headers[b_var_idx]
    b_var_cb = cj_coeffs[b_var_idx]
    
    cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.5, 1.5])
    cols[0].write(f"`{b_var_cb}`")
    cols[1].write(f"**{b_var_name}**")
    
    row_inputs = []
    for j in range(len(headers)):
        inp_val = cols[2 + j].number_input(f"R{i+1}_C{j+1}", value=0.0, step=0.1, key=f"cell_{step}_{i}_{j}", label_visibility="collapsed")
        row_inputs.append(inp_val)
        
    b_val_inp = cols[-2].number_input(f"R{i+1}_b", value=0.0, step=0.1, key=f"b_{step}_{i}", label_visibility="collapsed")
    
    # Calculate ratio if key column is known
    ratio_disp = "-"
    if iter_data['key_col'] != -1 and iter_data['key_col'] < len(headers):
        cv = expected_table[i, iter_data['key_col']]
        rv = expected_table[i, -1]
        ratio_disp = f"{rv/cv:.2f}" if cv > 1e-5 else "∞"
    cols[-1].write(f"`{ratio_disp}`")
    
    user_table_inputs.append(row_inputs + [b_val_inp])

# Render bottom Z-row input
z_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.5, 1.5])
z_cols[0].write("—")
z_cols[1].write("**Z-row**")
for j in range(len(headers)):
    z_inp = z_cols[2 + j].number_input(f"Z_C{j+1}", value=0.0, step=0.1, key=f"z_{step}_{j}", label_visibility="collapsed")
    user_z_inputs.append(z_inp)
z_val_inp = z_cols[-2].number_input(f"Z_b", value=0.0, step=0.1, key=f"z_b_{step}", label_visibility="collapsed")
z_cols[-1].write("—")

user_z_inputs.append(z_val_inp)

# --- VERIFICATION & ERROR HANDLING BUTTON ---
btn_verify = st.button("Finished Filling Table — Verify Calculations", type="primary")

if btn_verify:
    st.session_state.step_verified = False
    has_error = False
    error_log = []
    
    # Validate Core Row entries against Gauss-Jordan results
    for i in range(n_s):
        for j in range(len(headers) + 1):
            user_v = user_table_inputs[i][j]
            exp_v = expected_table[i, j]
            if not np.isclose(user_v, exp_v, atol=1e-1):
                has_error = True
                col_label = headers[j] if j < len(headers) else "b (RHS)"
                error_log.append((i, j, f"Row {i+1} ({headers[basis_indices[i]]}), Column '{col_label}': Expected `{exp_v:.2f}`, got `{user_v:.2f}`"))
                
    # Validate Z-row entries
    for j in range(len(headers) + 1):
        user_zv = user_z_inputs[j]
        exp_zv = expected_table[-1, j]
        if not np.isclose(user_zv, exp_zv, atol=1e-1):
            has_error = True
            col_label = headers[j] if j < len(headers) else "Z-Value"
            error_log.append((-1, j, f"Z-row, Column '{col_label}': Expected `{exp_zv:.2f}`, got `{user_zv:.2f}`"))

    if has_error:
        st.error("⚠️ Calculation Errors Detected in Tableau! Please correct the highlighted entries below:")
        for err in error_log:
            st.markdown(f"<div class='blink-error'>❌ {err[2]}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Excellent! All row reduction and tableau values are completely correct.")
        st.session_state.step_verified = True

# --- POST-VERIFICATION OPTIMALITY & PIVOT EVALUATION ---
if st.session_state.step_verified or iter_data['is_optimal']:
    st.markdown("---")
    
    if iter_data['is_optimal']:
        pt = iter_data['pt']
        max_z = iter_data['table'][-1, -1]
        st.balloons()
        st.markdown(f"""
        <div style='background-color: #1b4721; padding: 20px; border-radius: 10px; border: 2px solid #2ea043;'>
            <h3 style='color: #3fb950; margin:0;'>🎉 Optimality Condition Satisfied!</h3>
            <p style='font-size: 16px; color: #e6edf3; margin-top: 10px;'>
                All entries in the Z-row are non-negative (&ge; 0). No further improvement is possible.<br>
                <b>Optimal Decision Variables:</b> x<sub>1</sub> = {pt[0]:.2f}, x<sub>2</sub> = {pt[1]:.2f}<br>
                <b>Maximum Objective Z Value:</b> {max_z:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        k_col = iter_data['key_col']
        k_row = iter_data['key_row']
        entering_var = headers[k_col]
        leaving_var = headers[iter_data['basis'][k_row]]
        pivot_val = iter_data['table'][k_row, k_col]
        
        st.markdown("### Optimality & Feasibility Condition Analysis")
        st.warning("Optimality Condition Negative: Negative entries present in Z-row. Proceed to pivoting.")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Key Column (Entering Var)", entering_var, help="Most negative value in Z-row")
        col_p2.metric("Key Row (Leaving Var)", leaving_var, help="Minimum non-negative replacement ratio")
        col_p3.metric("Key Pivot Element", f"{pivot_val:.2f}", help="Intersection of Key Row and Key Column")
        col_p4.metric("Next Iteration Basis Replacement", f"{leaving_var} ➔ {entering_var}")
        
        if step < len(iterations) - 1:
            if st.button("Advance to Next Iteration Table", type="secondary"):
                st.session_state.current_step += 1
                st.session_state.step_verified = False
                st.rerun()

# --- GRAPHICAL VISUALIZATIONS (WITH AXES HIGHLIGHTING) ---
st.markdown("---")
st.markdown("### Geometric Feasible Region & Corner Trajectory")

tab1, tab2 = st.tabs(["2D Corner Trajectory & Highlighted Axes", "3D Objective Surface"])

curr_pt = iter_data['pt']

# Geometry helpers
def get_all_intersections(A, b_vec):
    lines = []
    for i in range(len(b_vec)):
        lines.append((A[i][0], A[i][1], b_vec[i]))
    lines.append((1.0, 0.0, 0.0)) # x1 = 0 axis line
    lines.append((0.0, 1.0, 0.0)) # x2 = 0 axis line
    
    points = []
    n_lines = len(lines)
    for i in range(n_lines):
        for j in range(i + 1, n_lines):
            a1, b1, c1_val = lines[i]
            a2, b2, c2_val = lines[j]
            det = a1 * b2 - a2 * b1
            if abs(det) > 1e-7:
                x1 = (c1_val * b2 - c2_val * b1) / det
                x2 = (a1 * c2_val - a2 * c1_val) / det
                points.append((x1, x2))
    return points

def is_feasible(pt, A, b_vec):
    x1, x2 = pt
    if x1 < -1e-5 or x2 < -1e-5:
        return False
    for i in range(len(b_vec)):
        if A[i][0] * x1 + A[i][1] * x2 > b_vec[i] + 1e-5:
            return False
    return True

def sort_polygon_vertices(points):
    pts = np.array(points)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]

with tab1:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    all_pts = get_all_intersections(A, b_vec)
    bfs_points, bs_points = [], []
    
    for p in all_pts:
        if -1 <= p[0] <= 10 and -1 <= p[1] <= 10:
            if is_feasible(p, A, b_vec):
                if not any(np.allclose(p, existing, atol=1e-4) for existing in bfs_points):
                    bfs_points.append(p)
            else:
                if not any(np.allclose(p, existing, atol=1e-4) for existing in bs_points):
                    bs_points.append(p)
                    
    # Fill Feasible Region
    if len(bfs_points) >= 3:
        sorted_bfs = sort_polygon_vertices(bfs_points)
        ax.fill(sorted_bfs[:, 0], sorted_bfs[:, 1], color='#2ea043', alpha=0.35, label='Feasible Region')
        closed_poly = np.vstack([sorted_bfs, sorted_bfs[0]])
        ax.plot(closed_poly[:, 0], closed_poly[:, 1], color='#3fb950', linewidth=3.5)

    # Constraint Boundary Lines
    x_grid = np.linspace(-0.5, 5.0, 400)
    for i in range(len(b_vec)):
        a1_val, a2_val = A[i]
        if abs(a2_val) > 1e-5:
            y_grid = (b_vec[i] - a1_val * x_grid) / a2_val
            ax.plot(x_grid, y_grid, label=f'C{i+1}: {a1_val}x1 + {a2_val}x2 <= {b_vec[i]}', linestyle='--', linewidth=1.8)
        else:
            ax.axvline(x=b_vec[i]/a1_val, label=f'C{i+1}: {a1_val}x1 <= {b_vec[i]}', linestyle='--', linewidth=1.8)

    # --- HIGHLIGHTED X1 & X2 COORDINATE AXES ---
    ax.axhline(0, color='#00d2ff', linewidth=2.5, zorder=3, label='Highlighted x1-axis (x2 = 0)')
    ax.axvline(0, color='#7ee787', linewidth=2.5, zorder=3, label='Highlighted x2-axis (x1 = 0)')

    # Basic Solutions vs Basic Feasible Solutions
    if bs_points:
        bs_arr = np.array(bs_points)
        ax.scatter(bs_arr[:, 0], bs_arr[:, 1], color='#f85149', s=90, marker='X', zorder=4, label='Basic Solution (Infeasible)')

    if bfs_points:
        bfs_arr = np.array(bfs_points)
        ax.scatter(bfs_arr[:, 0], bfs_arr[:, 1], color='#58a6ff', s=110, marker='o', zorder=5, label='Basic Feasible Solution (BFS)')

    # Path Vector Arrow
    if not iter_data['is_optimal'] and iter_data['key_row'] != -1:
        next_pt = iterations[step + 1]['pt'] if step + 1 < len(iterations) else curr_pt
        if next_pt != curr_pt:
            ax.annotate('', xy=(next_pt[0], next_pt[1]), xytext=(curr_pt[0], curr_pt[1]),
                         arrowprops=dict(facecolor='#d29922', edgecolor='#d29922', shrink=0.05, width=2.5, headwidth=9))

    # Current Point Indicator
    ax.scatter([curr_pt[0]], [curr_pt[1]], color='#f0883e', s=220, zorder=6, edgecolors='white', linewidth=2, label='Current Iteration Corner')

    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 4.0)
    ax.set_title("Corner Point Trajectory & Coordinate Axes", fontweight='bold', fontsize=14, color='#58a6ff')
    ax.set_xlabel("x1 (Decision Variable 1)", fontsize=12, color='#00d2ff')
    ax.set_ylabel("x2 (Decision Variable 2)", fontsize=12, color='#7ee787')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc='upper right')

    st.pyplot(fig)

with tab2:
    x1_a = np.linspace(0, 4, 30)
    x2_a = np.linspace(0, 4, 30)
    X1, X2 = np.meshgrid(x1_a, x2_a)
    Z = c1 * X1 + c2 * X2
    
    fig3d = go.Figure()
    fig3d.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='Viridis', opacity=0.6, showscale=False))
    
    path_pts = [it['pt'] for it in iterations[:step+1]]
    px = [p[0] for p in path_pts]
    py = [p[1] for p in path_pts]
    pz = [c1*x + c2*y for x, y in zip(px, py)]
    
    fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers+text',
                                marker=dict(size=8, color='#f38ba8'),
                                line=dict(color='#f38ba8', width=5),
                                text=[f"Iter {i}" for i in range(len(px))]))
    
    fig3d.update_layout(scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='Z Elevation'),
                        margin=dict(l=0, r=0, b=0, t=30), height=550, template="plotly_dark")
    st.plotly_chart(fig3d, use_container_width=True)
