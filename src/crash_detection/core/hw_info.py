import requests
import platform
import subprocess


def is_internet_connected():
    """Checks for an internet connection by attempting to reach Google."""
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False


def get_os_type():
    os_name = platform.system()
    if os_name == "Windows":
        return "Windows"
    elif os_name == "Darwin":  # Darwin is the kernel name for macOS
        return "macOS"
    elif os_name == "Linux":
        return "Linux"
    else:
        return f"Unknown OS: {os_name}"


def get_hw_details():
    os_name = get_os_type()
    match os_name:
        case "Windows":
            n_procs = subprocess.check_output("", shell=True).decode()
            n_threads = subprocess.check_output("", shell=True).decode()
        case "macOS":
            n_procs = subprocess.check_output(
                "sysctl -n hw.physicalcpu", shell=True
            ).decode()
            n_threads = subprocess.check_output(
                "sysctl -n hw.logicalcpu", shell=True
            ).decode()
        case "Linux":
            n_procs = subprocess.check_output(
                "lscpu | grep -Po '^Thread\(s\) per core:\s*\K\d+'",
                shell=True,
            ).decode()
            n_threads = subprocess.check_output(
                "lscpu | grep -Po '^Core\(s\) per socket:\s*\K\d+'",
                shell=True,
            ).decode()
        case _:
            n_procs = 1
            n_threads = 1

    if isinstance(n_procs, str):
        n_procs = int(n_procs.strip())

    if isinstance(n_threads, str):
        n_threads = int(n_threads.strip())

    if os_name == "macOS":
        n_threads //= n_procs

    return n_procs, n_threads
