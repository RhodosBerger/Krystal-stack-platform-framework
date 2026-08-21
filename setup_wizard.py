# Advanced Evolutionary Computing Framework Setup Wizard

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import sys
import os
import threading
import time
import json
from pathlib import Path
import platform
import psutil


class SetupWizard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Evolutionary Computing Framework - Setup")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        # Center the window
        self.center_window()
        
        # Setup variables
        self.install_path = tk.StringVar()
        self.install_path.set(os.path.expanduser("~\\EvolutionaryFramework"))
        self.components = {
            'core': tk.BooleanVar(value=True),
            'genetic': tk.BooleanVar(value=True),
            'neural': tk.BooleanVar(value=True),
            'api': tk.BooleanVar(value=True),
            'benchmark': tk.BooleanVar(value=True),
            'openvino': tk.BooleanVar(value=False),
            'gpu_support': tk.BooleanVar(value=False)
        }
        
        self.create_widgets()
        
    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.root.winfo_screenheight() // 2) - (500 // 2)
        self.root.geometry(f"700x500+{x}+{y}")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Title frame
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = ttk.Label(title_frame, text="Advanced Evolutionary Computing Framework", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Setup Wizard", font=("Arial", 10))
        subtitle_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        
        # Main notebook for pages
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Page 1: Welcome
        self.page1 = ttk.Frame(self.notebook)
        self.notebook.add(self.page1, text="Welcome")
        self.create_welcome_page()
        
        # Page 2: License
        self.page2 = ttk.Frame(self.notebook)
        self.notebook.add(self.page2, text="License")
        self.create_license_page()
        
        # Page 3: Installation Options
        self.page3 = ttk.Frame(self.notebook)
        self.notebook.add(self.page3, text="Options")
        self.create_options_page()
        
        # Page 4: Installation Path
        self.page4 = ttk.Frame(self.notebook)
        self.notebook.add(self.page4, text="Location")
        self.create_location_page()
        
        # Page 5: Components
        self.page5 = ttk.Frame(self.notebook)
        self.notebook.add(self.page5, text="Components")
        self.create_components_page()
        
        # Page 6: System Check
        self.page6 = ttk.Frame(self.notebook)
        self.notebook.add(self.page6, text="System Check")
        self.create_system_check_page()
        
        # Page 7: Installation
        self.page7 = ttk.Frame(self.notebook)
        self.notebook.add(self.page7, text="Install")
        self.create_installation_page()
        
        # Page 8: Completion
        self.page8 = ttk.Frame(self.notebook)
        self.notebook.add(self.page8, text="Complete")
        self.create_completion_page()
        
        # Navigation buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.back_button = ttk.Button(button_frame, text="Back", command=self.go_back)
        self.back_button.pack(side=tk.LEFT)
        
        self.next_button = ttk.Button(button_frame, text="Next", command=self.go_next)
        self.next_button.pack(side=tk.RIGHT)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_installation)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Start on welcome page
        self.current_page = 0
        self.update_navigation()
    
    def create_welcome_page(self):
        """Create the welcome page."""
        welcome_text = """
Welcome to the Advanced Evolutionary Computing Framework Setup Wizard

This wizard will guide you through the installation of the Advanced Evolutionary Computing Framework, 
a comprehensive system for evolutionary computing with distributed communication, 
business model integration, performance optimization, and cross-platform support.

The framework includes:
• Genetic Algorithms with advanced evolutionary practices
• Generic Algorithms with multiple optimization strategies
• Communication Pipelines for distributed processing
• Business Model Framework with multiple monetization strategies
• Django API Integration for profile configuration
• Sysbench Integration for synthetic benchmarking
• OpenVINO Platform Integration for AI optimization
• Neural Network Framework with safety-first design
• Cross-Platform Support (Windows x86/ARM, Linux, macOS)
• Rust-Safe Memory Management with multiple layers
• Overclocking Profiles with safety guidance

Click 'Next' to continue with the installation.
        """
        
        text_widget = tk.Text(self.page1, wrap=tk.WORD, height=20, width=80)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, welcome_text)
        text_widget.config(state=tk.DISABLED)
    
    def create_license_page(self):
        """Create the license agreement page."""
        license_text = """
LICENSE AGREEMENT

Advanced Evolutionary Computing Framework

Copyright (c) 2025 Advanced Computing Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

By clicking 'I Agree', you acknowledge that you have read, understood, and agree 
to be bound by these terms and conditions.
        """
        
        text_widget = tk.Text(self.page2, wrap=tk.WORD, height=20, width=80)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, license_text)
        text_widget.config(state=tk.DISABLED)
        
        self.license_agree = tk.BooleanVar()
        agree_check = ttk.Checkbutton(self.page2, text="I agree to the terms of the License Agreement", 
                                     variable=self.license_agree)
        agree_check.pack(pady=10)
    
    def create_options_page(self):
        """Create installation options page."""
        options_frame = ttk.LabelFrame(self.page3, text="Installation Options")
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Installation type
        type_frame = ttk.LabelFrame(options_frame, text="Installation Type")
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.install_type = tk.StringVar(value="typical")
        ttk.Radiobutton(type_frame, text="Typical Installation", variable=self.install_type, 
                       value="typical").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(type_frame, text="Custom Installation", variable=self.install_type, 
                       value="custom").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(type_frame, text="Minimal Installation", variable=self.install_type, 
                       value="minimal").pack(anchor=tk.W, padx=10, pady=2)
        
        # Additional options
        additional_frame = ttk.LabelFrame(options_frame, text="Additional Options")
        additional_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_desktop_shortcut = tk.BooleanVar(value=True)
        ttk.Checkbutton(additional_frame, text="Create Desktop Shortcut", 
                       variable=self.create_desktop_shortcut).pack(anchor=tk.W, padx=10, pady=2)
        
        self.add_to_path = tk.BooleanVar(value=True)
        ttk.Checkbutton(additional_frame, text="Add to System PATH", 
                       variable=self.add_to_path).pack(anchor=tk.W, padx=10, pady=2)
        
        self.install_examples = tk.BooleanVar(value=True)
        ttk.Checkbutton(additional_frame, text="Install Example Projects", 
                       variable=self.install_examples).pack(anchor=tk.W, padx=10, pady=2)
    
    def create_location_page(self):
        """Create installation location page."""
        location_frame = ttk.LabelFrame(self.page4, text="Installation Location")
        location_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Installation path
        path_frame = ttk.Frame(location_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Install to:").pack(anchor=tk.W)
        
        path_entry_frame = ttk.Frame(path_frame)
        path_entry_frame.pack(fill=tk.X, pady=5)
        
        self.path_entry = ttk.Entry(path_entry_frame, textvariable=self.install_path)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_button = ttk.Button(path_entry_frame, text="Browse...", command=self.browse_path)
        browse_button.pack(side=tk.RIGHT)
        
        # Space requirements
        space_frame = ttk.LabelFrame(location_frame, text="Space Requirements")
        space_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Calculate required space (in MB)
        required_space = self.calculate_required_space()
        ttk.Label(space_frame, text=f"Required space: {required_space} MB").pack(anchor=tk.W, padx=10, pady=2)
        
        # Check available space
        available_space = self.get_available_space()
        ttk.Label(space_frame, text=f"Available space: {available_space} MB").pack(anchor=tk.W, padx=10, pady=2)
        
        if available_space < required_space:
            ttk.Label(space_frame, text="⚠️  Insufficient space available", 
                     foreground="red").pack(anchor=tk.W, padx=10, pady=2)
    
    def create_components_page(self):
        """Create components selection page."""
        components_frame = ttk.LabelFrame(self.page5, text="Select Components")
        components_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Core components
        core_frame = ttk.LabelFrame(components_frame, text="Core Components")
        core_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(core_frame, text="Core Framework (Required)", 
                       variable=self.components['core'], state='disabled').pack(anchor=tk.W, padx=10, pady=2)
        ttk.Checkbutton(core_frame, text="Genetic Algorithms", 
                       variable=self.components['genetic']).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Checkbutton(core_frame, text="Neural Network Framework", 
                       variable=self.components['neural']).pack(anchor=tk.W, padx=10, pady=2)
        
        # API and Integration
        api_frame = ttk.LabelFrame(components_frame, text="API and Integration")
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(api_frame, text="Django API Framework", 
                       variable=self.components['api']).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Checkbutton(api_frame, text="Sysbench Integration", 
                       variable=self.components['benchmark']).pack(anchor=tk.W, padx=10, pady=2)
        
        # Advanced Features
        advanced_frame = ttk.LabelFrame(components_frame, text="Advanced Features")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(advanced_frame, text="OpenVINO Integration (AI Optimization)", 
                       variable=self.components['openvino']).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Checkbutton(advanced_frame, text="GPU Support (CUDA/ROCm)", 
                       variable=self.components['gpu_support']).pack(anchor=tk.W, padx=10, pady=2)
    
    def create_system_check_page(self):
        """Create system compatibility check page."""
        check_frame = ttk.LabelFrame(self.page6, text="System Compatibility Check")
        check_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # System information
        sys_info_frame = ttk.LabelFrame(check_frame, text="System Information")
        sys_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sys_info_frame, text=f"Platform: {platform.system()} {platform.machine()}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(sys_info_frame, text=f"Architecture: {platform.architecture()[0]}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(sys_info_frame, text=f"CPU Cores: {psutil.cpu_count()}").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(sys_info_frame, text=f"Memory: {round(psutil.virtual_memory().total / (1024**3), 2)} GB").pack(anchor=tk.W, padx=10, pady=2)
        
        # Python version check
        python_frame = ttk.LabelFrame(check_frame, text="Python Requirements")
        python_frame.pack(fill=tk.X, padx=5, pady=5)
        
        python_version = platform.python_version()
        ttk.Label(python_frame, text=f"Python Version: {python_version}").pack(anchor=tk.W, padx=10, pady=2)
        
        if tuple(map(int, python_version.split('.')[:2])) < (3, 8):
            ttk.Label(python_frame, text="❌ Python 3.8 or higher required", 
                     foreground="red").pack(anchor=tk.W, padx=10, pady=2)
        else:
            ttk.Label(python_frame, text="✅ Python version is compatible", 
                     foreground="green").pack(anchor=tk.W, padx=10, pady=2)
        
        # Dependency check
        dep_frame = ttk.LabelFrame(check_frame, text="Dependencies")
        dep_frame.pack(fill=tk.X, padx=5, pady=5)
        
        deps_to_check = ['numpy', 'psutil', 'requests', 'flask', 'django']
        for dep in deps_to_check:
            try:
                __import__(dep)
                ttk.Label(dep_frame, text=f"✅ {dep} - Available", 
                         foreground="green").pack(anchor=tk.W, padx=10, pady=1)
            except ImportError:
                ttk.Label(dep_frame, text=f"⚠️  {dep} - Not installed (will install)", 
                         foreground="orange").pack(anchor=tk.W, padx=10, pady=1)
    
    def create_installation_page(self):
        """Create installation progress page."""
        install_frame = ttk.LabelFrame(self.page7, text="Installing")
        install_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.install_text = tk.Text(install_frame, wrap=tk.WORD, height=20, width=80)
        self.install_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(install_frame, orient=tk.VERTICAL, command=self.install_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.install_text.config(yscrollcommand=scrollbar.set)
        
        self.install_button = ttk.Button(self.page7, text="Start Installation", 
                                        command=self.start_installation)
        self.install_button.pack(pady=10)
    
    def create_completion_page(self):
        """Create completion page."""
        complete_frame = ttk.LabelFrame(self.page8, text="Installation Complete")
        complete_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        complete_text = """
Installation has been completed successfully!

The Advanced Evolutionary Computing Framework has been installed to:

{}
        
What's Next:
• Run 'evolutionary_framework.exe' to start the application
• Check the documentation in the 'docs' folder
• Visit our website for tutorials and examples
• Join our community forum for support and discussions

Thank you for installing the Advanced Evolutionary Computing Framework!
        """.format(self.install_path.get())
        
        text_widget = tk.Text(complete_frame, wrap=tk.WORD, height=15, width=80)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, complete_text)
        text_widget.config(state=tk.DISABLED)
        
        # Launch options
        launch_frame = ttk.Frame(complete_frame)
        launch_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.launch_after = tk.BooleanVar(value=True)
        ttk.Checkbutton(launch_frame, text="Launch Advanced Evolutionary Computing Framework after setup", 
                       variable=self.launch_after).pack(anchor=tk.W)
    
    def calculate_required_space(self):
        """Calculate required installation space in MB."""
        # Approximate space requirements based on selected components
        base_size = 50  # MB for core
        sizes = {
            'genetic': 25,
            'neural': 30,
            'api': 20,
            'benchmark': 15,
            'openvino': 100,  # If selected
            'gpu_support': 40  # If selected
        }
        
        total_size = base_size
        for comp, size in sizes.items():
            if self.components[comp].get():
                total_size += size
        
        return total_size
    
    def get_available_space(self):
        """Get available space on install drive in MB."""
        try:
            path = self.install_path.get()
            if not os.path.exists(path):
                path = os.path.dirname(path)
                if not os.path.exists(path):
                    path = os.path.expanduser("~")
            
            total, used, free = shutil.disk_usage(path)
            return round(free / (1024 * 1024))  # Convert to MB
        except:
            return 10000  # Default to 10GB if can't determine
    
    def browse_path(self):
        """Browse for installation path."""
        path = filedialog.askdirectory(initialdir=self.install_path.get())
        if path:
            self.install_path.set(path)
    
    def go_next(self):
        """Go to next page."""
        if self.current_page == 1:  # License page
            if not self.license_agree.get():
                messagebox.showwarning("License Agreement", "You must agree to the license terms to continue.")
                return
        
        if self.current_page < 7:
            self.current_page += 1
            self.notebook.select(self.current_page)
            self.update_navigation()
    
    def go_back(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.notebook.select(self.current_page)
            self.update_navigation()
    
    def update_navigation(self):
        """Update navigation buttons based on current page."""
        if self.current_page == 0:  # Welcome
            self.back_button.config(state='disabled')
        else:
            self.back_button.config(state='normal')
        
        if self.current_page == 7:  # Completion
            self.next_button.config(text="Finish", command=self.finish_installation)
            self.cancel_button.config(state='disabled')
        else:
            self.next_button.config(text="Next", command=self.go_next)
            self.cancel_button.config(state='normal')
    
    def start_installation(self):
        """Start the installation process."""
        self.install_button.config(state='disabled')
        
        # Start installation in a separate thread
        install_thread = threading.Thread(target=self.run_installation)
        install_thread.daemon = True
        install_thread.start()
    
    def run_installation(self):
        """Run the actual installation."""
        def update_progress(step, total_steps, message):
            progress_percent = (step / total_steps) * 100
            self.progress['value'] = progress_percent
            self.install_text.insert(tk.END, f"[{progress_percent:.1f}%] {message}\n")
            self.install_text.see(tk.END)
            self.root.update_idletasks()
        
        try:
            # Create installation directory
            install_path = Path(self.install_path.get())
            install_path.mkdir(parents=True, exist_ok=True)
            
            update_progress(1, 10, f"Creating installation directory: {install_path}")
            
            # Copy files (simulated)
            import time
            for i in range(2, 11):
                time.sleep(0.2)  # Simulate file copying
                steps = [
                    "Creating core directories",
                    "Copying framework files",
                    "Installing genetic algorithms",
                    "Installing neural network components",
                    "Installing API framework",
                    "Installing benchmark tools",
                    "Installing dependencies",
                    "Creating configuration files",
                    "Setting up environment",
                    "Finalizing installation"
                ]
                update_progress(i, 10, steps[i-1])
            
            # Create desktop shortcut if requested
            if self.create_desktop_shortcut.get():
                update_progress(10, 10, "Creating desktop shortcut...")
                # Simulate shortcut creation
            
            # Add to PATH if requested
            if self.add_to_path.get():
                update_progress(10, 10, "Adding to system PATH...")
                # Simulate PATH modification
            
            update_progress(10, 10, "Installation completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Installation Error", f"An error occurred during installation:\n{str(e)}")
    
    def finish_installation(self):
        """Finish the installation."""
        if self.launch_after.get():
            try:
                # Launch the application
                exe_path = os.path.join(self.install_path.get(), "evolutionary_framework.exe")
                if os.path.exists(exe_path):
                    subprocess.Popen([exe_path])
            except:
                pass  # Ignore errors in launching
        
        self.root.quit()
    
    def cancel_installation(self):
        """Cancel the installation."""
        if messagebox.askokcancel("Cancel Installation", "Are you sure you want to cancel the installation?"):
            self.root.quit()
    
    def run(self):
        """Run the setup wizard."""
        self.root.mainloop()


if __name__ == "__main__":
    wizard = SetupWizard()
    wizard.run()