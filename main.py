import sys
import os
import argparse
import platform
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPolygon, PathPatch
from matplotlib.path import Path
from model_wrapper import MLQModel

MODEL_PATH = r"D:\mlq_desktop\best_model_3_Meta_raw_data.keras"
SCALER_X_PATH = r"D:\mlq_desktop\scaler_x.pkl"
SCALER_Y_PATH = r"D:\mlq_desktop\scaler_y.pkl"

def optimize_compute(model, freq, R, Lg, Ll, topk_percent=10):
    # User's optimization routine: 50 Tw samples and top-k threshold
    Tw_vals = np.linspace(0.1, 10.0, 100)
    # Vectorized prediction: model.predict_q to handle broadcasting
    Q_vals = model.predict_q(Tw_vals, freq, R, Lg, Ll)

    # Find best
    idx_max = int(np.nanargmax(Q_vals))
    Q_max = float(Q_vals[idx_max])
    best_tw = float(Tw_vals[idx_max])

    df = pd.DataFrame({'Tw [mm]': Tw_vals, 'Q': Q_vals})
    threshold = Q_max * (1 - float(topk_percent) / 100.0)
    top_k_df = df[df['Q'] >= threshold].copy()

    result_text = f"Best Design:\nTw = {best_tw:.4f} mm\nMax Q = {Q_max:.4f}\n"
    result_text += f"\nTop-{float(topk_percent):.0f}% Region Candidates: {len(top_k_df)}\n"
    # include a short preview (first few rows)
    result_text += top_k_df.head().to_string(index=False)

    return {
        'Tw_vals': Tw_vals,
        'Q_vals': Q_vals,
        'best_tw': best_tw,
        'Q_max': Q_max,
        'top_k_df': top_k_df,
        'result_text': result_text
    }
    

# Headless mode: save plots to PNG and print results
def run_headless(args):
    print('Running in headless mode (no GUI)')
    print('Python:', sys.executable)
    print('Platform:', platform.platform())
    model = MLQModel(MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH)

    freq = float(args.freq)
    R = float(args.R)
    Lg = float(args.Lg)
    Ll = float(args.Ll)
    topk = int(args.topk)

    res = optimize_compute(model, freq, R, Lg, Ll, topk)

    print(res['result_text'])

    # Ensure Agg backend (no GUI)
    matplotlib.use('Agg')

    # Plot 1: plot top-k region 
    fig1, ax = plt.subplots(figsize=(6, 4), dpi=150)
    top_k_df = res['top_k_df']
    if not top_k_df.empty:
        ax.plot(top_k_df['Tw [mm]'], top_k_df['Q'], marker='o', linestyle='-', color='blue')
    else:
        ax.plot(res['Tw_vals'], res['Q_vals'], marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Trace Width (Tw) [mm]')
    ax.set_ylabel('Q Factor')
    ax.set_title(f"Top-{float(topk):.0f}% Designs")
    ax.grid(True)
    out1 = os.path.abspath(args.out_prefix + '_q_vs_tw.png')
    fig1.tight_layout(pad=0.5)
    fig1.savefig(out1, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Q vs Frequency at best Tw
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
    freq_range = np.linspace(100.0, 800.0, 800)
    Q_f = model.predict_q(res['best_tw'], freq_range, R, Lg, Ll)
    ax2.plot(freq_range, Q_f, '-', label=f'Q vs Frequency @ Tw={res["best_tw"]:.3f}')
    ax2.axvline(freq, color='gray', linestyle='--', label='Input Frequency')
    ax2.plot([freq], [float(model.predict_q(res['best_tw'], freq, R, Lg, Ll))], 'o', color='blue', markersize=7, label='Optimized Point')
    ax2.set_xlabel('Frequency [MHz]')
    ax2.set_ylabel('Q Factor')
    ax2.set_title('Q vs Frequency')
    ax2.grid(True)
    ax2.legend()
    out2 = os.path.abspath(args.out_prefix + '_q_vs_freq.png')
    fig2.tight_layout(pad=0.5)
    fig2.savefig(out2, bbox_inches='tight')
    plt.close(fig2)

    # Coil diagram 
    fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=150)
    render_coil_axes(ax3, R, Lg, Ll, res['best_tw'])
    out3 = os.path.abspath(args.out_prefix + '_coil.png')
    fig3.tight_layout(pad=0.5)
    fig3.savefig(out3, bbox_inches='tight')
    plt.close(fig3)

    print('Saved plots:')
    print(' -', out1)
    print(' -', out2)
    print(' -', out3)


def render_coil_axes(ax, R, Lg, Ll, Tw):
    """coil geometry: complete circle on top + two separate vertical legs extending down.
   
    """
    # Leg positions: symmetric about x=0, separated by gap Lg
    x_left = -Lg / 2.0
    x_right = Lg / 2.0
    # Build coil path:
    # 1. Complete circle (centered at origin with radius R)
    n_circle = 400
    theta_circle = np.linspace(0, 2*np.pi, n_circle)
    circle_x = R * np.cos(theta_circle)
    circle_y = R * np.sin(theta_circle)

    # Notch and leg geometry 
    # notch = max(R*0.15, Tw*10)
    # leg_w = max(Tw*30, 2.5)
    notch_h = max(R * 0.15, Tw * 10.0)
    leg_w = max(Tw * 30.0, 2.5)

    # Y coordinate where legs attach to the circle (approx)
    if abs(x_left) < R:
        y_attach = -np.sqrt(max(R**2 - x_left**2, 0.0))
    else:
        y_attach = -R

    notch_top = y_attach
    notch_bottom = notch_top - notch_h

    pad = 0.2
    mask = ~((circle_x >= (x_left - leg_w/2.0 - pad)) & (circle_x <= (x_right + leg_w/2.0 + pad)) & (circle_y <= notch_top))

    circle_x_filtered = circle_x[mask]
    circle_y_filtered = circle_y[mask]

    # filtered circle outline
    line_width = max(Tw * 30, 2.0)
    ax.plot(circle_x_filtered, circle_y_filtered, color='#333333', linewidth=line_width, solid_capstyle='round',
            solid_joinstyle='round', zorder=2)

    # legs as filled rectangles whose TOP aligns with notch_top
    from matplotlib.patches import Rectangle
    left_rect = Rectangle((x_left - leg_w/2.0, notch_top - Ll), leg_w, Ll, facecolor='#333333', edgecolor='none', zorder=2)
    right_rect = Rectangle((x_right - leg_w/2.0, notch_top - Ll), leg_w, Ll, facecolor='#333333', edgecolor='none', zorder=2)
    ax.add_patch(left_rect)
    ax.add_patch(right_rect)
    cover_h = max(0.6, notch_h * 0.6)
    cover = Rectangle((x_left - leg_w, notch_top - cover_h/2.0), (x_right + leg_w) - (x_left - leg_w), cover_h,
                     facecolor='white', edgecolor='none', zorder=10)
    ax.add_patch(cover)
    ax.plot(circle_x_filtered, circle_y_filtered, color='black', linewidth=1.6, solid_capstyle='round',
            solid_joinstyle='round', zorder=4)

    # left leg sides
    ax.plot([x_left - leg_w/2.0, x_left - leg_w/2.0], [notch_top - Ll, notch_top - 0.01], color='black', linewidth=1.2, zorder=4)
    ax.plot([x_left + leg_w/2.0, x_left + leg_w/2.0], [notch_top - Ll, notch_top - 0.01], color='black', linewidth=1.2, zorder=4)

    # right leg sides
    ax.plot([x_right - leg_w/2.0, x_right - leg_w/2.0], [notch_top - Ll, notch_top - 0.01], color='black', linewidth=1.2, zorder=4)
    ax.plot([x_right + leg_w/2.0, x_right + leg_w/2.0], [notch_top - Ll, notch_top - 0.01], color='black', linewidth=1.2, zorder=4)
    
    # Dimension labels
    # R: radius label with dashed line to right side of circle (at top)
    ax.plot([0, R*0.85], [R*0.5, R*0.5], color='#FFD700', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.text(R*0.95, R*0.7, f'R={R:.1f}', fontsize=10, color='#FFD700', fontweight='bold', zorder=4)
    
    # Lg: gap label at bottom of legs with dimension line
    leg_bottom = -R - Ll
    ax.plot([x_left, x_right], [leg_bottom - 0.3, leg_bottom - 0.3], color='#A22B43', linewidth=1.2, alpha=0.7, zorder=1)
    ax.text(0, leg_bottom - 0.6, f'Lg={Lg:.1f} mm', fontsize=10, color='#A22B43', fontweight='bold', 
            ha='center', zorder=4)
    
    # Ll: leg length label on left side
    leg_mid = (-R + (-R - Ll)) / 2
    ax.text(x_left - 0.8, leg_mid, f'Ll={Ll:.1f} mm', fontsize=10, color='#4CAF50', fontweight='bold', 
            ha='right', va='center', zorder=4)
    
    # Tw: trace width label on right side
    ax.text(x_right + 0.8, leg_mid, f'Tw={Tw:.3f} mm', fontsize=10, color='#9C27B0', fontweight='bold', 
            ha='left', va='center', zorder=4)
    
    # Tw: trace width label on right side
    ax.text(x_right + 0.8, -Ll / 2, f'Tw={Tw:.3f} mm', fontsize=10, color='purple', fontweight='bold', 
            ha='left', va='center', zorder=4)
    
    # Layout
    ax.set_xlim(-R*1.3, R*1.4)
    ax.set_ylim(-Ll - 0.8, R + 0.5)
    ax.set_aspect('equal')
    ax.set_title('PCB Single Turn Tx Coil Geometry (mm)', fontsize=11, fontweight='bold')
    ax.axis('off')

# GUI mode
def run_gui(args):
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
    except Exception as e:
        print('Failed to import PyQt5/Qt backends:', e)
        raise
    
    # Force platform plugin
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    class PlotCanvas(FigureCanvas):
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = fig.add_subplot(111)
            super().__init__(fig)
            self.setParent(parent)
            fig.tight_layout()

    class MLQApp(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('MLQ: Meta Learning Based LPWPT System Tx Coil Geometry Optimization')
            self.model = MLQModel(MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH)
            self.init_ui()
            self.setMinimumSize(1100, 700)

        def init_ui(self):
            main_layout = QtWidgets.QHBoxLayout(self)
            main_layout.setContentsMargins(8, 8, 8, 8)
            main_layout.setSpacing(12)

            # Left panel
            left_panel = QtWidgets.QFrame()
            left_panel.setMinimumWidth(260)
            left_panel.setMaximumWidth(420)
            left_layout = QtWidgets.QVBoxLayout(left_panel)
            left_layout.setContentsMargins(16, 16, 16, 16)
            left_layout.setSpacing(12)

            title_label = QtWidgets.QLabel('<b>MLQ: Meta Learning Based LPWPT System Tx Coil Geometry Optimization</b>')
            title_label.setWordWrap(True)
            left_layout.addWidget(title_label)

            self.freq_input = QtWidgets.QDoubleSpinBox()
            self.freq_input.setRange(1.0, 10000.0)
            self.freq_input.setValue(400.0)
            self.freq_input.setSuffix(' MHz')
            left_layout.addWidget(QtWidgets.QLabel('Frequency (MHz)'))
            left_layout.addWidget(self.freq_input)

            self.r_input = QtWidgets.QDoubleSpinBox()
            self.r_input.setRange(0.1, 500.0)
            self.r_input.setValue(6.0)
            self.r_input.setSuffix(' mm')
            left_layout.addWidget(QtWidgets.QLabel('Outer Radius R (mm)'))
            left_layout.addWidget(self.r_input)

            self.lg_input = QtWidgets.QDoubleSpinBox()
            self.lg_input.setRange(0.0, 500.0)
            self.lg_input.setValue(5.0)
            self.lg_input.setSuffix(' mm')
            left_layout.addWidget(QtWidgets.QLabel('Coil leg gap Lg (mm)'))
            left_layout.addWidget(self.lg_input)

            self.ll_input = QtWidgets.QDoubleSpinBox()
            self.ll_input.setRange(0.1, 500.0)
            self.ll_input.setValue(10.0)
            self.ll_input.setSuffix(' mm')
            left_layout.addWidget(QtWidgets.QLabel('Coil leg length Ll (mm)'))
            left_layout.addWidget(self.ll_input)

            left_layout.addWidget(QtWidgets.QLabel('Top-k% Region'))
            self.topk_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.topk_slider.setRange(1, 50)
            self.topk_slider.setValue(10)
            left_layout.addWidget(self.topk_slider)
            self.topk_label = QtWidgets.QLabel('10 %')
            left_layout.addWidget(self.topk_label)
            self.topk_slider.valueChanged.connect(self.on_topk_changed)

            left_layout.addStretch(1)
            self.optimize_button = QtWidgets.QPushButton('Optimize')
            self.optimize_button.setFixedHeight(48)
            self.optimize_button.clicked.connect(self.on_optimize)
            left_layout.addWidget(self.optimize_button)
            
            right_panel = QtWidgets.QFrame()
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(8, 8, 8, 8)
            right_layout.setSpacing(12)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.addWidget(left_panel)

            result_box = QtWidgets.QGroupBox('Optimization Result')
            result_layout = QtWidgets.QVBoxLayout(result_box)
            self.result_text = QtWidgets.QTextEdit()
            self.result_text.setReadOnly(True)
            self.result_text.setFixedHeight(70)
            result_layout.addWidget(self.result_text)
            right_layout.addWidget(result_box, stretch=0)

            plots_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.plot1 = PlotCanvas(None, width=3.5, height=2.8, dpi=100)
            self.plot2 = PlotCanvas(None, width=3.5, height=2.8, dpi=100)
            self.plot1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.plot2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            plots_splitter.addWidget(self.plot1)
            plots_splitter.addWidget(self.plot2)
            plots_splitter.setStretchFactor(0, 1)
            plots_splitter.setStretchFactor(1, 1)

            btn_bar = QtWidgets.QWidget()
            btn_layout = QtWidgets.QHBoxLayout(btn_bar)
            btn_layout.setContentsMargins(4, 4, 4, 4)
            btn_layout.setSpacing(8)
            self.save_plot1_btn = QtWidgets.QPushButton('Save Plot 1')
            self.save_plot2_btn = QtWidgets.QPushButton('Save Plot 2')
            self.save_coil_btn = QtWidgets.QPushButton('Save Coil')
            btn_layout.addWidget(self.save_plot1_btn)
            btn_layout.addWidget(self.save_plot2_btn)
            btn_layout.addWidget(self.save_coil_btn)
            btn_layout.addStretch(1)

            top_widget = QtWidgets.QWidget()
            top_v = QtWidgets.QVBoxLayout(top_widget)
            top_v.setContentsMargins(0, 0, 0, 0)
            top_v.setSpacing(6)
            top_v.addWidget(plots_splitter, stretch=1)
            top_v.addWidget(btn_bar, stretch=0)

            
            self.coil_plot = PlotCanvas(None, width=4, height=2.4, dpi=100)
            self.coil_plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.coil_plot.setMinimumHeight(140)
            vertical_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            vertical_split.addWidget(top_widget)
            vertical_split.addWidget(self.coil_plot)
            vertical_split.setStretchFactor(0, 4)
            vertical_split.setStretchFactor(1, 1)

            right_layout.addWidget(vertical_split, stretch=1)

            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([320, 900])
            main_layout.addWidget(splitter, stretch=1)
            
            self.on_optimize()
            self.save_plot1_btn.clicked.connect(self.save_plot1)
            self.save_plot2_btn.clicked.connect(self.save_plot2)
            self.save_coil_btn.clicked.connect(self.save_coil)

        def on_topk_changed(self, v):
            self.topk_label.setText(f"{v} %")

        def on_optimize(self):
            freq = float(self.freq_input.value())
            R = float(self.r_input.value())
            Lg = float(self.lg_input.value())
            Ll = float(self.ll_input.value())
            topk_percent = int(self.topk_slider.value())

            res = optimize_compute(self.model, freq, R, Lg, Ll, topk_percent)

            self.result_text.setText(res['result_text'])

           
            ax = self.plot1.axes
            ax.clear()
            top_k_df = res['top_k_df']
            if not top_k_df.empty:
                ax.plot(top_k_df['Tw [mm]'], top_k_df['Q'], marker='o', linestyle='-', color='blue', linewidth=2, markersize=4, label='Top-k%')
            else:
                ax.plot(res['Tw_vals'], res['Q_vals'], marker='o', linestyle='-', color='blue', linewidth=2, markersize=4, label='All')
         
            ax.axvline(res['best_tw'], color='red', linestyle='--', linewidth=1.5, label='Optimal Tw')
            ax.set_xlabel('Trace Width (Tw) [mm]', fontsize=8)
            ax.set_ylabel('Q Factor', fontsize=8)
            ax.set_title('Q vs Trace Width', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=6, framealpha=0.6)
            self.plot1.figure.tight_layout()
            self.plot1.draw()

 
            ax2 = self.plot2.axes
            ax2.clear()
            freq_range = np.linspace(100.0, 700.0, 300)
            Q_f = self.model.predict_q(res['best_tw'], freq_range, R, Lg, Ll)
            ax2.plot(freq_range, Q_f, color='purple', linewidth=2.5, label=f'S-param @ Tw={res["best_tw"]:.3f}')
            ax2.axvline(freq, color='blue', linestyle='--', linewidth=1.5, label='Input Frequency')
            ax2.plot([freq], [self.model.predict_q(res['best_tw'], freq, R, Lg, Ll)], marker='D', markersize=8, color='red', zorder=5, label='Optimized Point')
            ax2.set_xlabel('Frequency (MHz)', fontsize=8)
            ax2.set_ylabel('Q Factor', fontsize=8)
            ax2.set_title('Q vs Frequency', fontsize=10, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=6, framealpha=0.6)
            self.plot2.figure.tight_layout()
            self.plot2.draw()

            self.draw_coil(R, Lg, Ll, res['best_tw'])

        def draw_coil(self, R, Lg, Ll, Tw):
            ax = self.coil_plot.axes
            ax.clear()
            render_coil_axes(ax, R, Lg, Ll, Tw)
            self.coil_plot.figure.tight_layout()
            self.coil_plot.draw()

        def save_figure_dialog(self, fig, default_name='plot.png'):
            opts = QtWidgets.QFileDialog.Options()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save figure', default_name, 'PNG Files (*.png);;All Files (*)', options=opts)
            if path:
                try:
                    fig.savefig(path)
                    QtWidgets.QMessageBox.information(self, 'Saved', f'Saved to: {path}')
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to save: {e}')

        def save_plot1(self):
            self.save_figure_dialog(self.plot1.figure, 'plot_q_vs_tw.png')

        def save_plot2(self):
            self.save_figure_dialog(self.plot2.figure, 'plot_q_vs_freq.png')

        def save_coil(self):
            self.save_figure_dialog(self.coil_plot.figure, 'coil.png')

    
    app = QtWidgets.QApplication(sys.argv)
    print('Created QApplication')
    window = MLQApp()
    print('Created MLQApp window')
    
    window.setWindowState(QtCore.Qt.WindowActive)
    window.show()
    window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    window.raise_()
    window.activateWindow()
    window.setFocus()
    print('Window shown and activated')
    print('Entering event loop...')
    sys.exit(app.exec_())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLQ optimization GUI or headless runner')
    parser.add_argument('--nogui', action='store_true', help='Run in headless mode and save plots to disk')
    parser.add_argument('--out-prefix', default='mlq_output', help='Output filename prefix for headless plots')
    parser.add_argument('--freq', default=400.0, help='Frequency (MHz) for headless run')
    parser.add_argument('--R', default=6.0, help='Outer radius R (mm)')
    parser.add_argument('--Lg', default=5.0, help='Coil leg gap Lg (mm)')
    parser.add_argument('--Ll', default=10.0, help='Coil leg length Ll (mm)')
    parser.add_argument('--topk', default=10, help='Top-k percent region')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    args = parser.parse_args()

    if args.debug:
        print('Debug: Python executable:', sys.executable)
        print('Debug: Platform:', platform.platform())
        try:
            import importlib
            qt_spec = importlib.util.find_spec('PyQt5')
            print('Debug: PyQt5 spec:', qt_spec)
        except Exception as e:
            print('Debug: PyQt5 check failed:', e)

    if args.nogui:
        run_headless(args)
    else:
        try:
            print('Launching GUI...')
            run_gui(args)
        except Exception as e:
            print(f'ERROR: GUI failed to start: {e}')
            print('This typically means:')
            print('  - You are running on a headless/remote environment without a display')
            print('  - Qt platform plugin is missing or incompatible')
            print('')
            print('To run in headless mode and save plots instead, use:')
            print('  python main.py --nogui')
            import traceback
            traceback.print_exc()
            sys.exit(1)
