import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class BeamformingInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("5G Beamforming Simulator")
        self.root.geometry("800x600")
        
        self.colors = {
            'bg': '#f0f0f0',
            'frame': '#ffffff',
            'button': '#4a90e2',
            'text': '#333333'
        }
        
        self.params = {
            "Array Configuration": {
                "num_antennas": 64,
                "antenna_spacing": 0.5
            },
            "Signal Parameters": {
                "snr_db": 10,
                "num_paths": 3
            },
            "Beamforming Parameters": {
                "technique": ["Conventional", "Adaptive", "Hybrid"],
                "num_rf_chains": 8
            }
        }
        
        self.widgets = {}
        self.create_input_ui()

    def create_input_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(
            main_container,
            text="5G Beamforming Simulator",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        for category, params in self.params.items():
            frame = ttk.LabelFrame(main_container, text=category)
            frame.pack(fill=tk.X, padx=10, pady=5)

            if category not in self.widgets:
                self.widgets[category] = {}

            for param_name, param_value in params.items():
                param_frame = ttk.Frame(frame)
                param_frame.pack(fill=tk.X, padx=5, pady=2)

                label = ttk.Label(
                    param_frame,
                    text=param_name.replace('_', ' ').title()
                )
                label.pack(side=tk.LEFT)

                if isinstance(param_value, list):
                    var = tk.StringVar(value=param_value[0])
                    dropdown = ttk.Combobox(
                        param_frame,
                        values=param_value,
                        textvariable=var,
                        state='readonly'
                    )
                    dropdown.pack(side=tk.RIGHT, padx=5)
                    self.widgets[category][param_name] = var
                else:
                    entry = ttk.Entry(param_frame)
                    entry.insert(0, str(param_value))
                    entry.pack(side=tk.RIGHT, padx=5)
                    self.widgets[category][param_name] = entry

        simulate_button = ttk.Button(
            main_container,
            text="Run Simulation",
            command=self.on_simulate
        )
        simulate_button.pack(pady=20)

    def validate_params(self):
        for category, params in self.widgets.items():
            for param_name, widget in params.items():
                if isinstance(widget, ttk.Entry):
                    try:
                        value = float(widget.get())
                        if value <= 0:
                            raise ValueError(f"{param_name} must be positive")
                    except ValueError as e:
                        messagebox.showerror("Error", f"Invalid value for {param_name}: {str(e)}")
                        return False
        return True

    def on_simulate(self):
        if not self.validate_params():
            return

        simulation_params = {}
        for category, params in self.widgets.items():
            simulation_params[category] = {}
            for param_name, widget in params.items():
                if isinstance(widget, tk.StringVar):
                    simulation_params[category][param_name] = widget.get()
                else:
                    simulation_params[category][param_name] = float(widget.get())

        simulation_window = tk.Toplevel(self.root)
        BeamformingSimulation(simulation_window, simulation_params, self)

class BeamformingSimulation:
    def __init__(self, root, params, input_app):
        self.root = root
        self.root.title("5G Beamforming Simulation Results")
        self.root.geometry("1000x800")  # Reduced size since we're showing fewer plots
        self.params = params
        self.input_app = input_app
        
        self.create_ui()
        self.run_simulation()

    def create_ui(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.fig = Figure(figsize=(12, 5))  # Adjusted size for two plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_container)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_panel = ttk.Frame(self.main_container)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        ttk.Button(control_panel, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=5)
        ttk.Button(control_panel, text="New Simulation", command=self.new_simulation).pack(fill=tk.X, pady=5)

        params_frame = ttk.LabelFrame(control_panel, text="Simulation Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True)

        self.param_text = scrolledtext.ScrolledText(params_frame, width=40, height=20)
        self.param_text.pack(fill=tk.BOTH, expand=True)
        self.param_text.insert(tk.END, self.format_parameters())
        self.param_text.configure(state='disabled')

    def conventional_beamforming(self, channel, angles):
        num_antennas = int(self.params["Array Configuration"]["num_antennas"])
        d = self.params["Array Configuration"]["antenna_spacing"]
        
        steering_vectors = np.exp(-1j * 2 * np.pi * d * np.arange(num_antennas)[:, np.newaxis] * 
                                np.sin(angles))
        
        channel = channel.reshape(-1, 1)
        beam_pattern = np.abs(np.dot(channel.conjugate().T, steering_vectors))
        return 20 * np.log10(beam_pattern / np.max(beam_pattern))

    def adaptive_beamforming(self, channel, interference, angles):
        num_antennas = int(self.params["Array Configuration"]["num_antennas"])
        noise_power = 10 ** (-self.params["Signal Parameters"]["snr_db"] / 10)
        
        channel = channel.reshape(-1, 1)
        interference = interference.reshape(-1, 1)
        
        R = interference @ interference.conjugate().T + noise_power * np.eye(num_antennas)
        
        steering_vectors = self.get_steering_vectors(angles)
        weights = []
        for sv in steering_vectors.T:
            sv = sv.reshape(-1, 1)
            w = np.linalg.solve(R, sv)
            w = w / (sv.conjugate().T @ w)
            weights.append(w)
        
        beam_pattern = np.abs([channel.conjugate().T @ w for w in weights])
        return 20 * np.log10(beam_pattern / np.max(beam_pattern))

    def hybrid_beamforming(self, channel, angles):
        num_antennas = int(self.params["Array Configuration"]["num_antennas"])
        num_rf = int(self.params["Beamforming Parameters"]["num_rf_chains"])
        
        channel = channel.reshape(-1, 1)
        
        analog_bf = np.exp(1j * np.random.uniform(0, 2*np.pi, (num_antennas, num_rf)))
        effective_channel = channel.conjugate().T @ analog_bf
        digital_bf = effective_channel.conjugate().T @ np.linalg.inv(effective_channel @ effective_channel.conjugate().T)
        
        steering_vectors = self.get_steering_vectors(angles)
        combined_bf = analog_bf @ digital_bf
        beam_pattern = np.abs(combined_bf.conjugate().T @ steering_vectors)
        return 20 * np.log10(beam_pattern / np.max(beam_pattern))

    def get_steering_vectors(self, angles):
        num_antennas = int(self.params["Array Configuration"]["num_antennas"])
        d = self.params["Array Configuration"]["antenna_spacing"]
        return np.exp(-1j * 2 * np.pi * d * np.arange(num_antennas)[:, np.newaxis] * 
                     np.sin(angles))

    def generate_channel(self):
        num_antennas = int(self.params["Array Configuration"]["num_antennas"])
        num_paths = int(self.params["Signal Parameters"]["num_paths"])
        
        angles = np.random.uniform(-np.pi/3, np.pi/3, num_paths)
        gains = (np.random.randn(num_paths) + 1j * np.random.randn(num_paths)) / np.sqrt(2)
        
        channel = np.zeros(num_antennas, dtype=complex)
        steering_vectors = self.get_steering_vectors(angles)
        
        for i in range(num_paths):
            channel += gains[i] * steering_vectors[:, i]
            
        return channel / np.sqrt(num_paths)

    def run_simulation(self):
        self.fig.clear()
        
        channel = self.generate_channel()
        interference = self.generate_channel()
        angles = np.linspace(-np.pi/2, np.pi/2, 360)
        
        technique = self.params["Beamforming Parameters"]["technique"]
        
        # Polar plot
        ax_polar = self.fig.add_subplot(121, projection='polar')
        # Cartesian plot
        ax_cart = self.fig.add_subplot(122)
        
        if technique == 'Conventional':
            pattern = self.conventional_beamforming(channel, angles)
            title = 'Conventional Beamforming'
        elif technique == 'Adaptive':
            pattern = self.adaptive_beamforming(channel, interference, angles)
            title = 'Adaptive Beamforming (MVDR)'
        else:
            pattern = self.hybrid_beamforming(channel, angles)
            title = 'Hybrid Beamforming'
        
        pattern = pattern.flatten()
        
        # Plot polar pattern
        ax_polar.plot(angles, pattern)
        ax_polar.set_title(f"{title} - Polar Plot")
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.grid(True)
        ax_polar.set_ylim([-40, 0])
        
        # Plot cartesian pattern
        ax_cart.plot(np.degrees(angles), pattern)
        ax_cart.set_title(f"{title} - Cartesian Plot")
        ax_cart.set_xlabel("Angle (degrees)")
        ax_cart.set_ylabel("Magnitude (dB)")
        ax_cart.grid(True)
        ax_cart.set_ylim([-40, 0])
        ax_cart.set_xlim([-90, 90])
        
        self.fig.tight_layout()
        self.canvas.draw()

    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", "Results exported successfully!")

    def new_simulation(self):
        self.root.destroy()

    def format_parameters(self):
        text = "Simulation Parameters:\n\n"
        for category, params in self.params.items():
            text += f"{category}:\n"
            for param_name, value in params.items():
                text += f"  {param_name}: {value}\n"
            text += "\n"
        return text

if __name__ == "__main__":
    root = tk.Tk()
    app = BeamformingInputApp(root)
    root.mainloop()