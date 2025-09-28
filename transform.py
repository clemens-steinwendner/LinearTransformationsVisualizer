import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


##helpers
def apply_matrix(A, P):
    return (A@P.T).T

def safe_float(val, default=None):
    """ to make sure it's a valid float"""
    try: 
        return float(val)
    except Exception:
        return default
    


##vis
class LinTransApp:
    def __init__(self, root):
        self.root = root
        root.title("Linear Transformations in R^2")

        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Matrix A (2×2):").grid(row=0, column=0, padx=(0,8), sticky="w")

        ##matrix entries
        self.a11 = ttk.Entry(ctrl, width=6); self.a12 = ttk.Entry(ctrl, width=6)
        self.a21 = ttk.Entry(ctrl, width=6); self.a22 = ttk.Entry(ctrl, width=6)


        self.a11.grid(row=0, column=1); self.a12.grid(row=0, column=2, padx=(4,10))
        self.a21.grid(row=1, column=1); self.a22.grid(row=1, column=2, padx=(4,10))

        self.a11.insert(0, "1"); self.a12.insert(0, "0")
        self.a21.insert(0, "0"); self.a22.insert(0, "1")

        ##buttons for transform reset & special matrices
        self.btn_transform = ttk.Button(ctrl, text="Transform", command=self.on_transform)
        self.btn_reset = ttk.Button(ctrl, text="Reset", command=self.on_reset)
        self.btn_transform.grid(row=0, column=3, padx=6, sticky="ew")
        self.btn_reset.grid(row=1, column=3, padx=6, sticky="ew")

        ops = ttk.Frame(ctrl)
        ops.grid(row=0, column=4, rowspan=2, padx=(12, 0), sticky="n")

        ttk.Label(ops, text="Scale:").grid(row=0, column=0, sticky="e", padx=(0,6))
        self.scale_entry = ttk.Entry(ops, width=8)
        self.scale_entry.insert(0, "1.0")
        self.scale_entry.grid(row=0, column=1, sticky="w")
        self.btn_scale = ttk.Button(ops, text="Scale", command=self.on_scale)
        self.btn_scale.grid(row=0, column=2, padx=6, sticky="ew")

        ttk.Label(ops, text="Rotate (°):").grid(row=1, column=0, sticky="e", padx=(0,6))
        self.rotate_entry = ttk.Entry(ops, width=8)
        self.rotate_entry.insert(0, "0")
        self.rotate_entry.grid(row=1, column=1, sticky="w")
        self.btn_rotate = ttk.Button(ops, text="Rotate", command=self.on_rotate)
        self.btn_rotate.grid(row=1, column=2, padx=6, sticky="ew")

        self.btn_shear  = ttk.Button(ops, text="Shear",  command=self.on_shear)
        self.btn_mirror = ttk.Button(ops, text="Mirror (y-Axis)", command=self.on_mirror)
        self.btn_shear.grid(row=2, column=2, padx = 6, sticky="ew")
        self.btn_mirror.grid(row=3, column=2, padx = 6, sticky="ew")

        ops.grid_columnconfigure(1, weight=1)
        ops.grid_columnconfigure(2, weight=0)

        ##init for the plot
        self.fig, self.ax = plt.subplots(figsize=(6,6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        

        self.anim = None
        self.xlim = (-5, 5)
        self.ylim = (-5, 5)

        self.grid_pts = self._make_grid_lines(-5, 5, step=1)
        self.square_pts = self._unit_square()
        self.basis = np.array([[0,0],[1,0],[0,0],[0,1]])

        self._anims = []

        ##unit vecs
        self.e1_curr = np.array([1, 0], dtype=float)
        self.e2_curr = np.array([0, 1], dtype=float)

        self._store_original_geometry()

        self._init_plot()

    def _unit_square(self):
        
        return np.array([[0,0],[1,0],[1,1],[0,1],[0,0]], dtype=float)
    
    def _make_grid_lines(self, lo, hi, step=1):
        """initialize the grid at start"""
        lines = []
        ticks = np.arange(lo, hi+1, step)
        for t in ticks:
            lines.append(np.array([[t, lo], [t, hi]], dtype=float))
            lines.append(np.array([[lo, t],[hi, t]], dtype=float))

        return lines
    
    def _store_original_geometry(self):
        self.grid_src = [l.copy() for l in self.grid_pts]
        self.square_src = self.square_pts.copy()
        self.basis_src = self.basis.copy()

        

    ## plot setup
    def _init_plot(self):
        self.ax.clear()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Linear Transformation in $\\mathbb{R}^2$')
        self.ax.grid(False)

        self.ax.axhline(0, linewidth=1)
        self.ax.axvline(0, linewidth=1)

        self.grid_lines = []

        for line in self.grid_src:
            (ln,) = self.ax.plot(line[:,0], line[:,1], linewidth=0.5, alpha=0.3)
            self.grid_lines.append(ln)

        (self.square_line,) = self.ax.plot(self.square_src[:,0], self.square_src[:,1], linewidth=2, label='Unit square')

        self.e1 = self.ax.arrow(0, 0, 1, 0, width=0.02, length_includes_head=True)
        self.e2 = self.ax.arrow(0, 0, 0, 1, width=0.02, length_includes_head=True)

        e1_proxy, = self.ax.plot([], [], linewidth=2, label='$e_1$')
        e2_proxy, = self.ax.plot([], [], linewidth=2, label='$e_2$')
        self.ax.legend(handles=[self.square_line, e1_proxy, e2_proxy], loc='upper left')

        self.fig.tight_layout()
        self.canvas.draw()


    def _update_basis_arrows(self, e1_vec, e2_vec):
        if self.e1: self.e1.remove()
        if self.e2: self.e2.remove()
        self.e1 = self.ax.arrow(0, 0, e1_vec[0], e1_vec[1], width=0.02, length_includes_head=True)
        self.e2 = self.ax.arrow(0, 0, e2_vec[0], e2_vec[1], width=0.02, length_includes_head=True)

    
    ##actions

    def on_transform(self):

        #try read
        A = self._read_matrix()
        if A is None:
            return
        
        #can't click during transform
        self.btn_transform.configure(state=tk.DISABLED)
        self.btn_reset.configure(state=tk.DISABLED)

        #capture before
        grid_src = [l.copy() for l in self.grid_src]
        square_src = self.square_src.copy()
        e1_src = self.e1_curr.copy()
        e2_src = self.e2_curr.copy()


        #calculate transformation
        grid_tgt = [apply_matrix(A, l) for l in grid_src]
        square_tgt = apply_matrix(A, square_src)
        e1_tgt = A@e1_src
        e2_tgt = A@e2_src


        #init for animation
        frames = 60
        duration_ms = 900
        interval = max(15, duration_ms // frames)


        #for smooth transition
        def ease_in_out(t):
            return 3*t**2 - 2*t**3
        

        ##every frame calculate the next and draw
        def update(frame):
            t = ease_in_out(frame / (frames-1))

            for ln, src, tgt in zip(self.grid_lines, grid_src, grid_tgt):
                P = (1-t)*src + t*tgt
                ln.set_data(P[:,0], P[:,1])

            P = (1-t)*square_src + t*square_tgt
            self.square_line.set_data(P[:,0], P[:,1])

            e1_vec = (1-t)*e1_src + t*e1_tgt
            e2_vec = (1-t)*e2_src + t*e2_tgt
            self._update_basis_arrows(e1_vec, e2_vec)

            self.canvas.draw_idle()
            return (*self.grid_lines, self.square_line)
        

        #start animation
        self.anim = FuncAnimation(self.fig, update, frames=frames, interval=interval, blit=False, repeat=False)
        self._anims.append(self.anim)
        self.canvas.draw()

        ## store transformed vectors and reactive buttons when done
        def on_anim_done():
            self.grid_src = [g.copy() for g in grid_tgt]
            self.square_src = square_tgt.copy()
            self.e1_curr = e1_tgt.copy()
            self.e2_curr = e2_tgt.copy()

            self.btn_transform.configure(state=tk.NORMAL)
            self.btn_reset.configure(state = tk.NORMAL)

        self.root.after(duration_ms + 100, on_anim_done)


    def on_reset(self):
        self.a11.delete(0, tk.END); self.a11.insert(0, "1")
        self.a12.delete(0, tk.END); self.a12.insert(0, "0")
        self.a21.delete(0, tk.END); self.a21.insert(0, "0")
        self.a22.delete(0, tk.END); self.a22.insert(0, "1")

        self._store_original_geometry()
        self.e1_curr = np.array([1, 0], dtype=float)
        self.e2_curr = np.array([0, 1], dtype=float)

        # Reset visuals to original
        self._init_plot()


    def _read_matrix(self):
        vals = [safe_float(self.a11.get()),
                safe_float(self.a12.get()),
                safe_float(self.a21.get()),
                safe_float(self.a22.get())]
        if any(v is None for v in vals):
            messagebox.showerror("Invalid matrix", "Please enter valid numbers for the 2×2 matrix.")
            return None
        A = np.array([[vals[0], vals[1]],[vals[2], vals[3]]], dtype=float)

        return A
    

    ### implementation of special matrices scale shear, rotate and mirror on y axis
    def on_scale(self):

        factor = self.scale_entry.get() if self.scale_entry.get() else 1

        self.a11.delete(0, tk.END); self.a12.delete(0, tk.END)
        self.a21.delete(0, tk.END); self.a22.delete(0, tk.END)

        self.a11.insert(0, f"{factor}"); self.a12.insert(0, "0")
        self.a21.insert(0, "0"); self.a22.insert(0, f"{factor}")

    def on_shear(self):
        
        shear_amt = 1

        self.a11.delete(0, tk.END); self.a12.delete(0, tk.END)
        self.a21.delete(0, tk.END); self.a22.delete(0, tk.END)

        self.a11.insert(0, "1"); self.a12.insert(0, f"{shear_amt}")
        self.a21.insert(0, "0"); self.a22.insert(0, "1")

        

    def on_rotate(self):

        rotate_amt = float(self.rotate_entry.get()) if self.rotate_entry.get() else 0.0

        angle = np.radians(rotate_amt)
        self.a11.delete(0, tk.END); self.a12.delete(0, tk.END)
        self.a21.delete(0, tk.END); self.a22.delete(0, tk.END)

        self.a11.insert(0, f"{np.cos(angle)}"); self.a12.insert(0, f"{-1*np.sin(angle)}")
        self.a21.insert(0, f"{np.sin(angle)}"); self.a22.insert(0, f"{np.cos(angle)}")



    def on_mirror(self):

        self.a11.delete(0, tk.END); self.a12.delete(0, tk.END)
        self.a21.delete(0, tk.END); self.a22.delete(0, tk.END)

        self.a11.insert(0, "-1"); self.a12.insert(0, "0")
        self.a21.insert(0, "0"); self.a22.insert(0, "1")






if __name__ == "__main__":
    root = tk.Tk()

    try:
        from tkinter import font as tkfont
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=10)
    except Exception:
        pass

    app = LinTransApp(root)
    root.mainloop()
