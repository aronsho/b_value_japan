import os
def check_killed_jobs(job_id_prefix:str, n: int, log_dir="."):
    """
    Scan SLURM log files (slurm-<job_id_prefix>_<i>.out) for killed jobs,
    and also warn about missing log files.
    Parameters
    ----------
    job_id_prefix : str
        The base job id (e.g. "41001228")
    n : int
        Number of job indices to check (1..n)
    log_dir : str
        Directory where log files are stored
    Returns
    -------
    killed_indices : list of int
        List of job indices that were killed
    missing_indices : list of int
        List of job indices with no log file found
    """
    killed_keywords = ["TIME LIMIT", "OUT OF MEMORY", "CANCELLED", "exceeded"]
    killed_indices = []
    missing_indices = []
    for i in range(1, n+1):
        log_file = os.path.join(log_dir, f"slurm-{job_id_prefix}_{i}.out")
        if not os.path.exists(log_file):
            print(f"⚠️ Missing log file: {log_file}")
            missing_indices.append(i)
            continue
        try:
            with open(log_file, "r") as f:
                content = f.read()
                if any(kw in content for kw in killed_keywords):
                    killed_indices.append(i)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    return killed_indices, missing_indices
# Example usage
if __name__ == "__main__":
    killed, missing = check_killed_jobs("43211310", 5, log_dir=".")
    print("Killed job indices:", killed)
    print("Missing job indices:", missing)
