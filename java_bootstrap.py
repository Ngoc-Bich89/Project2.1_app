import os
import subprocess

def ensure_java():
    try:
        java_home = subprocess.check_output(["readlink", "-f", "/usr/bin/java"]).decode().strip()
        java_home = "/".join(java_home.split("/")[:-2])
        os.environ["JAVA_HOME"] = java_home
        print(f"JAVA_HOME set to: {java_home}")
    except Exception as e:
        print("Could not set JAVA_HOME:", e)
