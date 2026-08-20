# Install dependencies for interactive UI and plotting
!pip install -q ipywidgets plotly matplotlib numpy

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Enable high-contrast dark theme styling
plt.style.use('dark_background' if 'dark_background' in plt.style.available else 'default')

# --- GLOBAL STATE ---
state = {
    'num_constraints': 2,
    'c1': 2.0, 'c2': 1.0,
    'A': [], 'b': [], 'ops': [],
    'iterations': [],
    'current_step': 0,
    'headers': []
}

# --- STYLING CONSTANTS ---
HEADER_HTML = """
<div style="background: linear-gradient(135deg, #181825 0%, #313244 100%); padding: 18px; border-radius: 12px; color: #cdd6f4; border: 1px solid #45475a; margin-bottom: 15px;">
    <h2 style="color: #89b4fa; margin:0; font-family: sans-serif;">Simplex Algorithm Visualizer (Vidyavardhini College Layout)</h2>
    <p style="margin:6px 0 0 0; color: #bac2de; font-size: 13px;">
        Interactive 2D/3D Geometry, Student Tableau Validation, Ratio Calculations, and Orthographic Projections.
    </p>
</div>
"""

def make_card(title, widget):
    return widgets.VBox([
        widgets.HTML(f"<div style='font-weight:bold; color:#f5e0dc; font-size:13px; margin-bottom:5px;'>{title}</div>"),
        widget
    ], layout=widgets.Layout(border='1px solid #45475a', padding='10px', border_radius='8px', margin='5px 0', background_color='#181825'))

# --- UI CONTROLS ---
num_const_input = widgets.BoundedIntText(value=2, min=1, max=6, step=1, description='Constraints:', layout=widgets.Layout(width='180px'))
btn_setup = widgets.Button(description="Configure Problem Grid", button_style='info', icon='th')

dynamic_box = widgets.VBox()
std_form_output = widgets.Output()
tableau_output = widgets.Output()
calc_details_output = widgets.Output()
viz_output = widgets.Output()

# --- STEP 1: DYNAMIC INPUT SETUP ---
def build_grid(b=None):
    n = num_const_input.value
    state['num_constraints'] = n
    
    # Objective Coefficients matching Max Z = 2x1 + 1x2
    c1_in = widgets.FloatText(value=2.0, description='Max Z =', layout=widgets.Layout(width='140px'))
    c1_lbl = widgets.Label(value='x1 +')
    c2_in = widgets.FloatText(value=1.0, layout=widgets.Layout(width='60px'))
    c2_lbl = widgets.Label(value='x2')
    obj_box = widgets.HBox([c1_in, c1_lbl, c2_in, c2_lbl])
    
    # Defaults matching your LPP constraints
    def_a = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    def_b = [2.0, 1.0, 10.0, 10.0]
    
    rows = []
    state['ui_rows'] = []
    for i in range(n):
        a1_val = def_a[i][0] if i < len(def_a) else 1.0
        a2_val = def_a[i][1] if i < len(def_a) else 1.0
        b_val = def_b[i] if i < len(def_b) else 5.0
        
        a1 = widgets.FloatText(value=a1_val, layout=widgets.Layout(width='60px'))
        l1 = widgets.Label(value='x1 +')
        a2 = widgets.FloatText(value=a2_val, layout=widgets.Layout(width='60px'))
        l2 = widgets.Label(value='x2')
        op = widgets.Dropdown(options=['<=', '>='], value='<=', layout=widgets.Layout(width='65px'))
        rhs = widgets.FloatText(value=b_val, layout=widgets.Layout(width='70px'))
        
        row_widget = widgets.HBox([widgets.Label(value=f'C{i+1}:'), a1, l1, a2, l2, op, rhs])
        rows.append(row_widget)
        state['ui_rows'].append((a1, a2, op, rhs))
        
    btn_start = widgets.Button(description="Convert & Run Simplex", button_style='success', icon='play')
    btn_start.on_click(run_simplex_engine)
    
    state['ui_obj'] = (c1_in, c2_in)
    dynamic_box.children = [
        make_card("1. Objective Function (Maximization)", obj_box),
        make_card("2. Constraint Equations", widgets.VBox(rows)),
        btn_start
    ]

btn_setup.on_click(build_grid)

# --- STEP 2 & 3: STANDARDIZATION & SIMPLEX SOLVER ---
def run_simplex_engine(b=None):
    state['c1'] = state['ui_obj'][0].value
    state['c2'] = state['ui_obj'][1].value
    
    A, b_vec = [], []
    for a1, a2, op, rhs in state['ui_rows']:
        mult = -1.0 if op.value == '>=' else 1.0
        A.append([a1.value * mult, a2.value * mult])
        b_vec.append(rhs.value * mult)
        
    state['A'] = np.array(A)
    state['b'] = np.array(b_vec)
    n_s = len(b_vec)
    state['headers'] = ['x1', 'x2'] + [f's{i+1}' for i in range(n_s)]
    
    # Render Standard Form
    with std_form_output:
        clear_output()
        sf = f"<b>Standard Form Notation:</b><br>"
        sf += f"Maximize z - {state['c1']}x<sub>1</sub> - {state['c2']}x<sub>2</sub>" + "".join([f" - 0s<sub>{i+1}</sub>" for i in range(n_s)]) + " = 0<br>"
        sf += "Subject to constraints:<br>"
        for i in range(n_s):
            slacks = "".join([f" + 1s<sub>{j+1}</sub>" if i==j else f" + 0s<sub>{j+1}</sub>" for j in range(n_s)])
            sf += f"&nbsp;&nbsp;{state['A'][i][0]}x<sub>1</sub> + {state['A'][i][1]}x<sub>2</sub>{slacks} = {state['b'][i]}<br>"
        sf += f"&nbsp;&nbsp;x<sub>1</sub>, x<sub>2</sub>, s<sub>1</sub>..s<sub>{n_s}</sub> &ge; 0"
        display(HTML(f"<div style='background:#181825; padding:12px; border-radius:8px; border-left:4px solid #89b4fa;'>{sf}</div>"))

    # Compute Simplex Iteration Tables
    c_obj = np.array([-state['c1'], -state['c2']] + [0.0]*n_s)
    
    table = np.zeros((n_s + 1, 2 + n_s + 1))
    table[:-1, :2] = state['A']
    table[:-1, 2:2+n_s] = np.eye(n_s)
    table[:-1, -1] = state['b']
    table[-1, :-1] = c_obj
    
    basis = [2 + i for i in range(n_s)]
    state['iterations'] = []
    
    for it in range(10):
        z_row = table[-1, :-1]
        
        # Calculate current corner point (x1, x2)
        curr_pt = [0.0, 0.0]
        for idx, b_var in enumerate(basis):
            if b_var < 2:
                curr_pt[b_var] = table[idx, -1]
                
        # Check Optimality
        if np.all(z_row >= -1e-5):
            state['iterations'].append({
                'table': table.copy(), 'basis': list(basis), 'z_row': z_row.copy(),
                'key_col': -1, 'key_row': -1, 'ratios': [], 'is_optimal': True, 'pt': curr_pt
            })
            break
            
        key_col = int(np.argmin(z_row))
        
        # Ratio Calculation
        col_vals = table[:-1, key_col]
        rhs_vals = table[:-1, -1]
        ratios = []
        for cv, rv in zip(col_vals, rhs_vals):
            if cv > 1e-5:
                ratios.append(rv / cv)
            else:
                ratios.append(np.inf)
                
        key_row = int(np.argmin(ratios)) if any(r < np.inf for r in ratios) else -1
        
        state['iterations'].append({
            'table': table.copy(), 'basis': list(basis), 'z_row': z_row.copy(),
            'key_col': key_col, 'key_row': key_row, 'ratios': ratios, 'is_optimal': False, 'pt': curr_pt
        })
        
        if key_row == -1:
            break
            
        # Pivot Operation
        pivot = table[key_row, key_col]
        table[key_row, :] /= pivot
        for r in range(n_s + 1):
            if r != key_row:
                table[r, :] -= table[r, key_col] * table[key_row, :]
                
        basis[key_row] = key_col

    state['current_step'] = 0
    render_student_tableau_ui()
    render_graphics()

# --- STEP 4 & 5: STUDENT TABLEAU & ERROR CHECKING ---
def render_student_tableau_ui():
    step = state['current_step']
    iter_data = state['iterations'][step]
    headers = state['headers']
    
    with tableau_output:
        clear_output()
        display(HTML(f"<h3 style='color:#a6e3a1; margin-bottom:5px;'>Iteration {step} Simplex Table Evaluation</h3>"))
        
        if iter_data['is_optimal']:
            pt = iter_data['pt']
            max_z = iter_data['table'][-1, -1]
            display(HTML(f"""
            <div style='background:#181825; padding:15px; border-radius:8px; border-left:6px solid #a6e3a1;'>
                <h4 style='color:#a6e3a1; margin:0;'>OPTIMAL SOLUTION REACHED (All z-row entries &ge; 0)</h4>
                <p style='margin:5px 0 0 0; color:#cdd6f4;'>
                    <b>Optimal Corner Point:</b> x<sub>1</sub> = {pt[0]:.2f}, x<sub>2</sub> = {pt[1]:.2f} | <b>Max Z = {max_z:.2f}</b>
                </p>
            </div>
            """))
            with calc_details_output:
                clear_output()
            return

        # Interactive Row Inputs for Student
        display(HTML("<b>Fill up the z-row entries to evaluate optimality:</b>"))
        inputs = []
        for h in headers:
            w = widgets.FloatText(value=0.0, description=f'{h}:', layout=widgets.Layout(width='110px'))
            inputs.append(w)
            
        btn_check = widgets.Button(description="Verify Table & Proceed", button_style='primary', icon='check')
        
        def on_verify(b):
            user_vals = [w.value for w in inputs]
            expected = iter_data['z_row']
            
            with calc_details_output:
                clear_output()
                if np.allclose(user_vals, expected, atol=1e-1):
                    display(HTML("""
                    <div style='background:#313244; color:#a6e3a1; padding:8px; border-radius:5px;'>
                        <b>Verification Passed!</b> Solution is non-optimal. Key column and key row calculated below.
                    </div>
                    """))
                    show_step_details(iter_data, headers)
                else:
                    err_html = "<div style='background:#313244; color:#f38ba8; padding:8px; border-radius:5px;'><b>Wrong entries detected!</b> Re-check negative entries in z-row:<ul>"
                    for idx, (uv, ev) in enumerate(zip(user_vals, expected)):
                        if not np.isclose(uv, ev, atol=1e-1):
                            err_html += f"<li>Column <b>{headers[idx]}</b> expected {ev:.2f}, got {uv:.2f}</li>"
                    err_html += "</ul></div>"
                    display(HTML(err_html))
                    
        btn_check.on_click(on_verify)
        display(widgets.VBox([widgets.HBox(inputs), btn_check]))

def show_step_details(iter_data, headers):
    k_col = iter_data['key_col']
    k_row = iter_data['key_row']
    entering_var = headers[k_col]
    leaving_var = headers[iter_data['basis'][k_row]]
    
    r_str = "<br>".join([f"&nbsp;&nbsp;Row {i+1} ({headers[iter_data['basis'][i]]}): {r:.2f}" if r < np.inf else f"&nbsp;&nbsp;Row {i+1}: &infin;" for i, r in enumerate(iter_data['ratios'])])
    
    html = f"""
    <div style='background:#181825; padding:12px; border-radius:8px; border:1px solid #45475a; margin-top:10px;'>
        <b style='color:#89b4fa;'>Live Calculation & Variable Selection:</b><br>
        • <span style='color:#f9e2af;'>Key Column (Entering Variable):</span> <b>{entering_var}</b> (Most negative entry in z-row)<br>
        • <span style='color:#f38ba8;'>Key Row (Leaving Variable):</span> <b>{leaving_var}</b> (Least positive replacement ratio)<br>
        <br><b>Ratio Column Calculations:</b><br>{r_str}
    </div>
    """
    display(HTML(html))
    
    btn_next = widgets.Button(description="Advance to Next Iteration", button_style='success', icon='arrow-right')
    def advance(b):
        state['current_step'] += 1
        render_student_tableau_ui()
        render_graphics()
    btn_next.on_click(advance)
    display(btn_next)

# --- STEP 6: GRAPHICAL VISUALIZATION (2D, 3D & ENGINEERING PROJECTIONS) ---
def render_graphics():
    with viz_output:
        clear_output(wait=True)
        curr_step = state['current_step']
        iter_data = state['iterations'][min(curr_step, len(state['iterations'])-1)]
        curr_pt = iter_data['pt']
        
        c1, c2 = state['c1'], state['c2']
        A, b_vec = state['A'], state['b']
        
        # 3D Objective Surface & Path
        x1_a = np.linspace(0, 3, 25)
        x2_a = np.linspace(0, 3, 25)
        X1, X2 = np.meshgrid(x1_a, x2_a)
        Z = c1 * X1 + c2 * X2
        
        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='Viridis', opacity=0.55, showscale=False))
        
        path_pts = [it['pt'] for it in state['iterations'][:curr_step+1]]
        px = [p[0] for p in path_pts]
        py = [p[1] for p in path_pts]
        pz = [c1*x + c2*y for x, y in zip(px, py)]
        
        fig3d.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers+text',
                                    marker=dict(size=8, color='#f38ba8'),
                                    line=dict(color='#f38ba8', width=5),
                                    text=[f"Iter {i}" for i in range(len(px))]))
        
        fig3d.update_layout(title="3D Objective Elevation Surface (Z Plane)",
                            scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='Z Value'),
                            margin=dict(l=0, r=0, b=0, t=30), height=380, template="plotly_dark")
        fig3d.show()

        # Matplotlib 2D & Engineering Projections
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 2D Corner Point Trajectory
        ax1 = axes[0]
        x_grid = np.linspace(0, 3, 300)
        for i in range(len(b_vec)):
            a1, a2 = A[i]
            if a2 != 0:
                y_grid = (b_vec[i] - a1*x_grid) / a2
                ax1.plot(x_grid, y_grid, label=f'C{i+1}: {a1}x1+{a2}x2<={b_vec[i]}', linestyle='--')
            else:
                ax1.axvline(x=b_vec[i]/a1, label=f'C{i+1}: {a1}x1<={b_vec[i]}', linestyle='--')
                
        if not iter_data['is_optimal'] and iter_data['key_row'] != -1:
            next_pt = state['iterations'][curr_step + 1]['pt'] if curr_step + 1 < len(state['iterations']) else curr_pt
            if next_pt != curr_pt:
                ax1.annotate('', xy=(next_pt[0], next_pt[1]), xytext=(curr_pt[0], curr_pt[1]),
                             arrowprops=dict(facecolor='#fab387', edgecolor='#fab387', shrink=0.08, width=3, headwidth=10))

        ax1.scatter([curr_pt[0]], [curr_pt[1]], color='#f38ba8', s=140, zorder=5, label='Current Point')
        ax1.set_xlim(-0.2, 3.0)
        ax1.set_ylim(-0.2, 3.0)
        ax1.set_title("2D Simplex Corner Point Path", fontweight='bold', color='#89b4fa')
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='upper right')

        # Top Engineering Projection
        ax2 = axes[1]
        ax2.scatter([curr_pt[0]], [curr_pt[1]], color='#89dceb', s=120)
        ax2.set_title("Top View (x1 vs x2)", fontweight='bold', color='#89dceb')
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.set_xlim(-0.2, 3.0)
        ax2.set_ylim(-0.2, 3.0)
        ax2.grid(True, alpha=0.3)

        # Side Projection
        ax3 = axes[2]
        curr_z = c1*curr_pt[0] + c2*curr_pt[1]
        ax3.plot(px, pz, 'm--o', label='Trajectory Path')
        ax3.scatter([curr_pt[0]], [curr_z], color='#a6e3a1', s=140, label='Current Z')
        ax3.set_title("Side View (x1 vs Objective Z)", fontweight='bold', color='#a6e3a1')
        ax3.set_xlabel("x1")
        ax3.set_ylabel("Z Value")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

# Render Application UI
display(HTML(HEADER_HTML))
display(make_card("Configure Number of Constraints", widgets.HBox([num_const_input, btn_setup])))
display(dynamic_box)
display(std_form_output)
display(tableau_output)
display(calc_details_output)
display(viz_output)

# Initialize grid with defaults matching your LPP
build_grid()
