import os, glob
def _candidate_paths():
    candidates = ["/usr/lib/jvm/java-11-openjdk-amd64"]
    candidates += sorted(glob.glob("/usr/lib/jvm/*-openjdk-*"))
    return [p for p in candidates if os.path.exists(p)]
def ensure_java():
    if "JAVA_HOME" in os.environ and os.path.exists(os.environ["JAVA_HOME"]):
        return os.environ["JAVA_HOME"]
    for p in _candidate_paths():
        bin_path = os.path.join(p, "bin")
        if os.path.exists(bin_path):
            os.environ["JAVA_HOME"] = p
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH","")
            return p
    return None
