import launch

if not launch.is_installed("lora"):
    launch.run_pip("install git+https://github.com/cloneofsimo/lora.git", desc='Installing lora')

if not launch.is_installed("fire"):
    launch.run_pip("install fire", desc='Installing fire')
