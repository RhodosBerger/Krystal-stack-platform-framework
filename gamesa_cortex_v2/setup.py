from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="gamesa-cortex-v2",
    version="0.1.0",
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "gamesa_cortex_v2.rust_planner",
            "gamesa_cortex_v2/rust_planner/Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
    install_requires=[
        "numpy",
        "psutil",
        "openvino>=2023.0.0; platform_machine=='x86_64'", # Only install if available via pip
    ],
    author="Gamesa Cortex Team",
    author_email="dev@gamesacortex.com",
    description="The Neural Control Plane for Industry 5.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    zip_safe=False,
    python_requires=">=3.8",
)
