import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

# Imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Set page configuration
st.set_page_config(page_title="Simplex Practice & Visualizer", layout="wide")

# Custom CSS for bordered tables, blinking errors, and high-contrast visuals
st.markdown("""
<style>
    .main-title { font-size: 26px; font-weight: bold; color: #89b4fa; }
    .card { background-color: #1e1e2e; padding: 15px; border-radius: 10px; border: 1px solid #45475a; margin-bottom: 15px; }
    
    /* Blinking error styling */
    @keyframes blinker {
        50% { opacity: 0.2; background-color: #f85149; }
    }
    .blink-error {
        border: 2px solid #f85149 !important;
        border-radius: 5px;
        animation: blinker 1s linear infinite;
        padding: 8px;
        color: #ff7b72;
        font-weight: bold;
        margin-bottom: 6px;
    }
    
    /* Bordered Simplex Tableau Container */
    .tableau-box {
        border: 2px solid #89b4fa;
        border-radius: 8px;
        padding: 12px;
        background-color: #181825;
        margin-bottom: 20px;
    }
    .dotted-divider {
        border-top: 2px dashed #89b4fa;
        margin-top: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Simplex Method Practice & Interactive Visualizer</div>", unsafe_allow_html=True)
st.caption("Bordered Tableau Layout | 0 th Iteration Indexing | PDF Export & Lock Controls")

# --- SESSION STATE INITIALIZATION ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'step_verified' not in st.session_state:
    st.session_state.step_verified = False
if 'num_constraints' not in st.session_state:
    st.session_state.num_constraints = 2
if 'c1' not in st.session_state:
    st.session_state.c1 = 2.0
if 'c2' not in st.session_state:
    st.session_state.c2 = 1.0

# Initialize default constraint keys if not present
default_A_init = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
default_b_init = [2.0, 1.0, 5.0, 5.0]

for i in range(4):
    if f"a1_{i}" not in st.session_state:
        st.session_state[f"a1_{i}"] = default_A_init[i][0]
    if f"a2_{i}" not in st.session_state:
        st.session_state[f"a2_{i}"] = default_A_init[i][1]
    if f"op_{i}" not in st.session_state:
        st.session_state[f"op_{i}"] = "<="
    if f"rhs_{i}" not in st.session_state:
        st.session_state[f"rhs_{i}"] = default_b_init[i]

# --- SIDEBAR: PROBLEM FORMULATION & LOCK/RESET ---
st.sidebar.header("1. Problem Formulation")

col_sb1, col_sb2 = st.sidebar.columns(2)
lock_lpp = col_sb1.toggle("🔒 Lock LPP", value=False)

if col_sb2.button("🔄 Reset LPP"):
    st.session_state.num_constraints = 2
    st.session_state.c1 = 2.0
    st.session_state.c2 = 1.0
    for i in range(4):
        st.session_state[f"a1_{i}"] = default_A_init[i][0]
        st.session_state[f"a2_{i}"] = default_A_init[i][1]
        st.session_state[f"op_{i}"] = "<="
        st.session_state[f"rhs_{i}"] = default_b_init[i]
    st.session_state.current_step = 0
    st.session_state.step_verified = False
    st.rerun()

num_constraints = st.sidebar.number_input("Number of Constraints", min_value=1, max_value=4, key="num_constraints", disabled=lock_lpp, step=1)

st.sidebar.subheader("Objective Function (Max Z)")
col_c1, col_c2 = st.sidebar.columns(2)
c1 = col_c1.number_input("c1 (for x1)", key="c1", disabled=lock_lpp, step=0.5)
c2 = col_c2.number_input("c2 (for x2)", key="c2", disabled=lock_lpp, step=0.5)

st.sidebar.subheader("Constraints")
A_inputs, b_inputs, ops = [], [], []

for i in range(int(num_constraints)):
    st.sidebar.markdown(f"**Constraint {i+1}**")
    ca1, ca2, cop, crhs = st.sidebar.columns([2, 2, 2, 2])
    
    a1_val = ca1.number_input(f"x1 (C{i+1})", key=f"a1_{i}", disabled=lock_lpp)
    a2_val = ca2.number_input(f"x2 (C{i+1})", key=f"a2_{i}", disabled=lock_lpp)
    op_val = cop.selectbox(f"Op (C{i+1})", ["<=", ">="], key=f"op_{i}", disabled=lock_lpp)
    rhs_val = crhs.number_input(f"RHS (C{i+1})", key=f"rhs_{i}", disabled=lock_lpp)
    
    mult = -1.0 if op_val == ">=" else 1.0
    A_inputs.append([a1_val * mult, a2_val * mult])
    b_inputs.append(rhs_val * mult)
    ops.append(op_val)

A = np.array(A_inputs)
b_vec = np.array(b_inputs)
n_s = len(b_vec)
headers = ['x1', 'x2'] + [f's{i+1}' for i in range(n_s)]

# --- SIMPLEX ALGORITHM ENGINE ---
def compute_all_simplex_iterations(c1, c2, A, b_vec):
    n_s = len(b_vec)
    c_obj = np.array([-c1, -c2] + [0.0] * n_s)
    
    table = np.zeros((n_s + 1, 2 + n_s + 1))
    table[:-1, :2] = A
    table[:-1, 2:2+n_s] = np.eye(n_s)
    table[:-1, -1] = b_vec
    table[-1, :-1] = c_obj
    
    basis = [2 + i for i in range(n_s)] # Slack indices
    iterations = []
    
    for _ in range(10):
        z_row = table[-1, :-1]
        
        curr_pt = [0.0, 0.0]
        for idx, b_var in enumerate(basis):
            if b_var < 2:
                curr_pt[b_var] = table[idx, -1]
                
        # Check Optimality
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

step = st.session_state.current_step
iter_data = iterations[min(step, len(iterations)-1)]

# Text Label for Iteration Number
iter_title = "0 th Iteration" if step == 0 else f"{step} st Iteration" if step == 1 else f"{step} nd Iteration" if step == 2 else f"{step} th Iteration"

# --- SIMPLEX TABLE PRACTICE SECTION ---
st.markdown(f"### Simplex Table ({iter_title})")

col_tbl1, col_tbl2 = st.columns([1, 4])
lock_table = col_tbl1.toggle("🔒 Lock Table Inputs", value=False, key=f"lock_tbl_{step}")

if col_tbl1.button("🔄 Reset Table Inputs", key=f"reset_tbl_{step}"):
    for j in range(len(headers)):
        st.session_state[f"z_{step}_{j}"] = 0.0
    st.session_state[f"z_b_{step}"] = 0.0
    for i in range(n_s):
        for j in range(len(headers)):
            st.session_state[f"cell_{step}_{i}_{j}"] = 0.0
        st.session_state[f"b_{step}_{i}"] = 0.0
    st.session_state.step_verified = False
    st.rerun()

st.markdown("<div class='tableau-box'>", unsafe_allow_html=True)

# Main Column Headers
header_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
header_cols[0].markdown("**Iteration Number**")
header_cols[1].markdown("**Basic Variables**")
for idx, h in enumerate(headers):
    header_cols[2 + idx].markdown(f"**{h}**")
header_cols[-2].markdown("**R.H.S. Solution**")
header_cols[-1].markdown("**Ratio**")

# Z Row
z_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
z_cols[0].markdown(f"**{step}**")
z_cols[1].markdown("**z**")

user_z_inputs = []
for j in range(len(headers)):
    z_inp = z_cols[2 + j].number_input(f"Z_C{j+1}", value=st.session_state.get(f"z_{step}_{j}", 0.0), step=0.1, key=f"z_{step}_{j}", label_visibility="collapsed", disabled=lock_table)
    user_z_inputs.append(z_inp)

z_val_inp = z_cols[-2].number_input(f"Z_b", value=st.session_state.get(f"z_b_{step}", 0.0), step=0.1, key=f"z_b_{step}", label_visibility="collapsed", disabled=lock_table)
user_z_inputs.append(z_val_inp)
z_cols[-1].markdown("—")

st.markdown("<div class='dotted-divider'></div>", unsafe_allow_html=True)

# Basic Variables Rows
user_table_inputs = []
expected_table = iter_data['table']
basis_indices = iter_data['basis']

for i in range(n_s):
    b_var_idx = basis_indices[i]
    b_var_name = headers[b_var_idx]
    
    cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
    cols[0].write("")
    cols[1].markdown(f"**{b_var_name}**")
    
    row_inputs = []
    for j in range(len(headers)):
        inp_val = cols[2 + j].number_input(f"R{i+1}_C{j+1}", value=st.session_state.get(f"cell_{step}_{i}_{j}", 0.0), step=0.1, key=f"cell_{step}_{i}_{j}", label_visibility="collapsed", disabled=lock_table)
        row_inputs.append(inp_val)
        
    b_val_inp = cols[-2].number_input(f"R{i+1}_b", value=st.session_state.get(f"b_{step}_{i}", 0.0), step=0.1, key=f"b_{step}_{i}", label_visibility="collapsed", disabled=lock_table)
    
    if st.session_state.step_verified and not iter_data['is_optimal']:
        if iter_data['key_col'] != -1 and iter_data['key_col'] < len(headers):
            cv = expected_table[i, iter_data['key_col']]
            rv = expected_table[i, -1]
            ratio_disp = f"{rv/cv:.2f}" if cv > 1e-5 else "∞"
            cols[-1].markdown(f"**{ratio_disp}**")
        else:
            cols[-1].markdown("—")
    else:
        cols[-1].markdown("—")
        
    user_table_inputs.append(row_inputs + [b_val_inp])

st.markdown("</div>", unsafe_allow_html=True)

# --- VERIFICATION BUTTON & ERROR HANDLING ---
btn_verify = st.button("I am finished with filling up the table", type="primary")

if btn_verify:
    st.session_state.step_verified = False
    has_error = False
    error_log = []
    
    # Verify Z-row values
    for j in range(len(headers) + 1):
        user_zv = user_z_inputs[j]
        exp_zv = expected_table[-1, j]
        if not np.isclose(user_zv, exp_zv, atol=1e-1):
            has_error = True
            col_label = headers[j] if j < len(headers) else "R.H.S. Solution"
            error_log.append((-1, j, f"Z-row, Column '{col_label}': Expected `{exp_zv:.2f}`, got `{user_zv:.2f}`"))

    # Verify Core basic variable rows
    for i in range(n_s):
        for j in range(len(headers) + 1):
            user_v = user_table_inputs[i][j]
            exp_v = expected_table[i, j]
            if not np.isclose(user_v, exp_v, atol=1e-1):
                has_error = True
                col_label = headers[j] if j < len(headers) else "R.H.S. Solution"
                error_log.append((i, j, f"Row ({headers[basis_indices[i]]}), Column '{col_label}': Expected `{exp_v:.2f}`, got `{user_v:.2f}`"))

    if has_error:
        st.error("⚠️ Mistakes detected in tableau inputs! Check errors below:")
        for err in error_log:
            st.markdown(f"<div class='blink-error'>❌ {err[2]}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Table correctly filled! Ratios and pivot conditions are calculated below.")
        st.session_state.step_verified = True
        st.rerun()

# --- PDF GENERATOR HELPER ---
def generate_pdf_solution(iterations, headers, c1, c2, n_s):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    story = []
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1e1e2e'), spaceAfter=12)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=11, textColor=colors.HexColor('#45475a'), spaceAfter=18)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2ea043'), spaceBefore=12, spaceAfter=8)
    
    story.append(Paragraph("Simplex Method Solution Report", title_style))
    story.append(Paragraph(f"Objective Function: Max Z = {c1}x1 + {c2}x2 | Total Iterations: {len(iterations)-1}", sub_style))
    
    for idx, it in enumerate(iterations):
        it_label = "0 th Iteration" if idx == 0 else f"{idx} st Iteration" if idx == 1 else f"{idx} nd Iteration" if idx == 2 else f"{idx} th Iteration"
        story.append(Paragraph(f"<b>{it_label}</b>", h2_style))
        
        table_data = []
        # Header Row
        table_data.append(["Iter #", "Basic Var"] + headers + ["R.H.S.", "Ratio"])
        
        # Z Row
        z_row_vals = [f"{val:.2f}" for val in it['table'][-1, :-1]] + [f"{it['table'][-1, -1]:.2f}", "—"]
        table_data.append([str(idx), "z"] + z_row_vals)
        
        # Basic Rows
        for r_idx in range(n_s):
            b_var_name = headers[it['basis'][r_idx]]
            r_vals = [f"{val:.2f}" for val in it['table'][r_idx, :-1]] + [f"{it['table'][r_idx, -1]:.2f}"]
            
            ratio_str = "—"
            if not it['is_optimal'] and it['key_col'] != -1:
                cv = it['table'][r_idx, it['key_col']]
                rv = it['table'][r_idx, -1]
                ratio_str = f"{rv/cv:.2f}" if cv > 1e-5 else "∞"
                
            table_data.append(["", b_var_name] + r_vals + [ratio_str])
            
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#89b4fa')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cdd6f4')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f2cdcd')),
        ]))
        
        story.append(t)
        story.append(Spacer(1, 12))
        
    # Optimal Result Summary
    opt_it = iterations[-1]
    opt_pt = opt_it['pt']
    opt_z = opt_it['table'][-1, -1]
    
    story.append(Paragraph("<b>Final Optimal Solution</b>", h2_style))
    summary_text = f"Optimal Point: x1 = {opt_pt[0]:.2f}, x2 = {opt_pt[1]:.2f} <br/> Maximum Z = {opt_z:.2f}"
    story.append(Paragraph(summary_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- OPTIMALITY CONDITION, PIVOT DISPLAY & ALL TABLES VIEW ---
if st.session_state.step_verified:
    st.markdown("---")
    
    if iter_data['is_optimal']:
        pt = iter_data['pt']
        max_z = iter_data['table'][-1, -1]
        st.balloons()
        st.markdown(f"""
        <div style='background-color: #1b4721; padding: 20px; border-radius: 10px; border: 2px solid #2ea043;'>
            <h3 style='color: #3fb950; margin:0;'>🎉 Optimality Condition Satisfied!</h3>
            <p style='font-size: 16px; color: #e6edf3; margin-top: 10px;'>
                Since all coefficients in the objective equation (Z-row) are non-negative (&ge; 0), the solution is optimal.<br>
                <b>Optimal Coordinates:</b> x<sub>1</sub> = {pt[0]:.2f}, x<sub>2</sub> = {pt[1]:.2f}<br>
                <b>Maximum Objective Value Z:</b> {max_z:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- ALL SIMPLEX TABLES AT A GLANCE & PDF DOWNLOAD ---
        st.markdown("### 📊 Complete Solution: All Simplex Tables at a Glance")
        
        # Download PDF Button
        pdf_data = generate_pdf_solution(iterations, headers, c1, c2, n_s)
        st.download_button(
            label="📥 Download Complete Solution PDF Report",
            data=pdf_data,
            file_name="Simplex_Solution_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        # Display all iteration tables sequentially
        for idx, it in enumerate(iterations):
            it_lbl = "0 th Iteration" if idx == 0 else f"{idx} st Iteration" if idx == 1 else f"{idx} nd Iteration" if idx == 2 else f"{idx} th Iteration"
            with st.expander(f"📌 {it_lbl} Table View", expanded=True):
                tbl = it['table']
                
                # Header
                h_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
                h_cols[0].markdown("**Iter #**")
                h_cols[1].markdown("**Basic Var**")
                for j_idx, h in enumerate(headers):
                    h_cols[2 + j_idx].markdown(f"**{h}**")
                h_cols[-2].markdown("**R.H.S.**")
                h_cols[-1].markdown("**Ratio**")
                
                # Z row
                z_c = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
                z_c[0].write(str(idx))
                z_c[1].write("z")
                for j_idx in range(len(headers)):
                    z_c[2 + j_idx].write(f"{tbl[-1, j_idx]:.2f}")
                z_c[-2].write(f"{tbl[-1, -1]:.2f}")
                z_c[-1].write("—")
                
                st.markdown("<div class='dotted-divider'></div>", unsafe_allow_html=True)
                
                # Basic Rows
                for r_idx in range(n_s):
                    r_cols = st.columns([1.5, 1.5] + [1.2] * len(headers) + [1.8, 1.5])
                    r_cols[0].write("")
                    r_cols[1].write(headers[it['basis'][r_idx]])
                    for j_idx in range(len(headers)):
                        r_cols[2 + j_idx].write(f"{tbl[r_idx, j_idx]:.2f}")
                    r_cols[-2].write(f"{tbl[r_idx, -1]:.2f}")
                    
                    ratio_str = "—"
                    if not it['is_optimal'] and it['key_col'] != -1:
                        cv = tbl[r_idx, it['key_col']]
                        rv = tbl[r_idx, -1]
                        ratio_str = f"{rv/cv:.2f}" if cv > 1e-5 else "∞"
                    r_cols[-1].write(ratio_str)

    else:
        k_col = iter_data['key_col']
        k_row = iter_data['key_row']
        entering_var = headers[k_col]
        leaving_var = headers[iter_data['basis'][k_row]]
        pivot_val = iter_data['table'][k_row, k_col]
        
        st.markdown("### Optimality & Feasibility Condition Results")
        st.warning("Negative coefficients present in Z-row. Optimality condition NOT satisfied.")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Key Column (Entering Var)", entering_var)
        col_p2.metric("Key Row (Leaving Var)", leaving_var)
        col_p3.metric("Key Element (Pivot)", f"{pivot_val:.2f}")
        col_p4.metric("Basis Variable Replacement", f"{leaving_var} ➔ {entering_var}")
        
        if step < len(iterations) - 1:
            next_step_num = step + 1
            next_label = f"{next_step_num} st Iteration" if next_step_num == 1 else f"{next_step_num} nd Iteration" if next_step_num == 2 else f"{next_step_num} th Iteration"
            if st.button(f"Form New Table for {next_label} ➔", type="secondary"):
                st.session_state.current_step += 1
                st.session_state.step_verified = False
                st.rerun()

# --- GRAPHICAL VISUALIZATION (2D & 3D) ---
st.markdown("---")
st.markdown("### Geometry & Coordinate Trajectory View")

tab1, tab2 = st.tabs(["2D Corner Trajectory & Feasible Region", "3D Objective Surface & Base Projection"])

curr_pt = iter_data['pt']

def get_all_intersections(A, b_vec):
    lines = []
    for i in range(len(b_vec)):
        lines.append((A[i][0], A[i][1], b_vec[i]))
    lines.append((1.0, 0.0, 0.0))
    lines.append((0.0, 1.0, 0.0))
    
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

# --- TAB 1: INTERACTIVE 2D PLOT WITH HOVER TOOLTIPS ---
with tab1:
    fig2d = go.Figure()

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

    # 1. Feasible Region Polygon Area
    if len(bfs_points) >= 3:
        sorted_bfs = sort_polygon_vertices(bfs_points)
        poly_x = list(sorted_bfs[:, 0]) + [sorted_bfs[0, 0]]
        poly_y = list(sorted_bfs[:, 1]) + [sorted_bfs[0, 1]]
        
        fig2d.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            fill="toself",
            fillcolor="rgba(46, 160, 67, 0.35)",
            line=dict(color="#3fb950", width=3),
            name="Feasible Region",
            hoverinfo="skip"
        ))

    # 2. Constraint Boundary Lines
    x_line = np.linspace(-0.5, 5.0, 200)
    line_colors = ['#f5e0dc', '#cba6f7', '#f38ba8', '#fab387']
    for i in range(len(b_vec)):
        a1_val, a2_val = A[i]
        color = line_colors[i % len(line_colors)]
        if abs(a2_val) > 1e-5:
            y_line = (b_vec[i] - a1_val * x_line) / a2_val
            fig2d.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                name=f'C{i+1}: {a1_val}x1 + {a2_val}x2 <= {b_vec[i]}',
                hoverinfo="skip"
            ))
        else:
            fig2d.add_trace(go.Scatter(
                x=[b_vec[i]/a1_val]*2, y=[-0.5, 5.0], mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                name=f'C{i+1}: {a1_val}x1 <= {b_vec[i]}',
                hoverinfo="skip"
            ))

    # 3. Infeasible Basic Solution Intersection Points (Hover Enabled)
    if bs_points:
        bs_x = [p[0] for p in bs_points]
        bs_y = [p[1] for p in bs_points]
        bs_z = [c1 * p[0] + c2 * p[1] for p in bs_points]
        bs_hover = [f"<b>Infeasible Basic Solution</b><br>x1: {p[0]:.2f}<br>x2: {p[1]:.2f}<br>Z: {z:.2f}" for p, z in zip(bs_points, bs_z)]
        
        fig2d.add_trace(go.Scatter(
            x=bs_x, y=bs_y, mode='markers',
            marker=dict(size=12, color='#f85149', symbol='x'),
            name='Infeasible Corner Point',
            hovertext=bs_hover,
            hoverinfo='text'
        ))

    # 4. Feasible Basic Corner Points (BFS) (Hover Enabled)
    if bfs_points:
        bfs_x = [p[0] for p in bfs_points]
        bfs_y = [p[1] for p in bfs_points]
        bfs_z = [c1 * p[0] + c2 * p[1] for p in bfs_points]
        bfs_hover = [f"<b>Basic Feasible Solution (BFS)</b><br>x1: {p[0]:.2f}<br>x2: {p[1]:.2f}<br>Z: {z:.2f}" for p, z in zip(bfs_points, bfs_z)]
        
        fig2d.add_trace(go.Scatter(
            x=bfs_x, y=bfs_y, mode='markers',
            marker=dict(size=14, color='#89b4fa', symbol='circle'),
            name='BFS Corner Point',
            hovertext=bfs_hover,
            hoverinfo='text'
        ))

    # 5. Full Trajectory Path Trace
    full_path_pts = [it['pt'] for it in iterations[:step+1]]
    px = [p[0] for p in full_path_pts]
    py = [p[1] for p in full_path_pts]
    pz = [c1 * p[0] + c2 * p[1] for p in full_path_pts]
    path_hover = [f"<b>Iteration {i} Path Point</b><br>x1: {p[0]:.2f}<br>x2: {p[1]:.2f}<br>Z: {z:.2f}" for i, (p, z) in enumerate(zip(full_path_pts, pz))]

    fig2d.add_trace(go.Scatter(
        x=px, y=py, mode='lines+markers+text',
        line=dict(color='#fab387', width=3),
        marker=dict(size=10, color='#f9e2af'),
        text=[f"Iter {i}" for i in range(len(px))],
        textposition="top right",
        name='Simplex Trajectory',
        hovertext=path_hover,
        hoverinfo='text'
    ))

    # 6. Highlight Current Active Iteration Point
    curr_z = c1 * curr_pt[0] + c2 * curr_pt[1]
    curr_hover = f"<b>CURRENT POINT (Iter {step})</b><br>x1: {curr_pt[0]:.2f}<br>x2: {curr_pt[1]:.2f}<br>Z: {curr_z:.2f}"
    
    fig2d.add_trace(go.Scatter(
        x=[curr_pt[0]], y=[curr_pt[1]], mode='markers',
        marker=dict(size=18, color='#ff007f', line=dict(color='white', width=2)),
        name=f'Current Point (Iter {step})',
        hovertext=[curr_hover],
        hoverinfo='text'
    ))

    # Layout Formatting
    fig2d.update_layout(
        title="Interactive 2D Feasible Region & Corner Trajectory",
        xaxis=dict(title="x1 (Decision Variable 1)", range=[-0.5, 4.0], zeroline=True, zerolinecolor='#00f5ff', zerolinewidth=2),
        yaxis=dict(title="x2 (Decision Variable 2)", range=[-0.5, 4.0], zeroline=True, zerolinecolor='#39ff14', zerolinewidth=2),
        height=600,
        template="plotly_dark",
        hoverlabel=dict(bgcolor="#181825", font_size=13, font_family="Arial")
    )

    st.plotly_chart(fig2d, use_container_width=True)

# --- TAB 2: 3D OBJECTIVE SURFACE WITH BASE PLANE PROJECTION ---
with tab2:
    x1_a = np.linspace(0, 4, 30)
    x2_a = np.linspace(0, 4, 30)
    X1, X2 = np.meshgrid(x1_a, x2_a)
    Z = c1 * X1 + c2 * X2
    
    fig3d = go.Figure()
    
    # 1. 3D Surface for Objective Function
    fig3d.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='Viridis', opacity=0.55, showscale=False, name="Objective Surface"))
    
    # 2. Feasible Region Projection on x1-x2 Base Plane (z=0)
    if len(bfs_points) >= 3:
        sorted_bfs = sort_polygon_vertices(bfs_points)
        bx = list(sorted_bfs[:, 0]) + [sorted_bfs[0, 0]]
        by = list(sorted_bfs[:, 1]) + [sorted_bfs[0, 1]]
        bz = [0.0] * len(bx)
        
        fig3d.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='lines',
            line=dict(color='#3fb950', width=6),
            name="Base Plane Feasible Region"
        ))
        
        fig3d.add_trace(go.Mesh3d(
            x=sorted_bfs[:, 0], y=sorted_bfs[:, 1], z=np.zeros(len(sorted_bfs)),
            color='#2ea043', opacity=0.4, name="Base Feasible Area"
        ))

    # 3. Trajectory Path on 3D Objective Surface
    path_pts = [it['pt'] for it in iterations[:step+1]]
    px = [p[0] for p in path_pts]
    py = [p[1] for p in path_pts]
    pz = [c1*x + c2*y for x, y in zip(px, py)]
    
    fig3d.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines+markers+text',
        marker=dict(size=8, color='#ff007f'),
        line=dict(color='#ff007f', width=6),
        text=[f"Iter {i}" for i in range(len(px))],
        textposition="top center",
        name="Objective Path"
    ))
    
    # 4. Perpendicular Drop Line from Objective Surface to Base Plane
    opt_x1, opt_x2 = curr_pt[0], curr_pt[1]
    opt_z = c1 * opt_x1 + c2 * opt_x2
    
    fig3d.add_trace(go.Scatter3d(
        x=[opt_x1, opt_x1],
        y=[opt_x2, opt_x2],
        z=[0, opt_z],
        mode='lines+markers',
        line=dict(color='#f9e2af', width=7, dash='dash'),
        marker=dict(size=7, color=['#39ff14', '#ff007f']),
        name="Z Drop-Line to Base"
    ))
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title='x1 (Base Plane)',
            yaxis_title='x2 (Base Plane)',
            zaxis_title='Z (Objective Altitude)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=580,
        template="plotly_dark"
    )
    st.plotly_chart(fig3d, use_container_width=True)
