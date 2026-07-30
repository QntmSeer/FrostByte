"""
Remote Workstation Benchmark Runner
===================================
Utility script to connect to the remote workstation (NVIDIA RTX A2000 GPU) via Paramiko,
transfer core model files and scripts, and execute workstation benchmarks.
"""

import paramiko
import os
import sys

HOST = "192.168.1.112"
USER = "ikar"
PASS = "1011bin"
REMOTE_DIR = "/home/ikar/diffusion-cryoem-prior"

def main():
    print(f"Connecting to remote workstation {USER}@{HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(HOST, username=USER, password=PASS, timeout=10)
        print("Connected successfully!")
        
        py_path = "/home/ikar/protein_design/venv/bin/python3"
        
        # Sync required code files
        ssh.exec_command(f"mkdir -p {REMOTE_DIR}/scripts {REMOTE_DIR}/models {REMOTE_DIR}/utils")
        sftp = ssh.open_sftp()
        local_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        files_to_sync = [
            ("scripts/benchmark_a2000_workstation.py", f"{REMOTE_DIR}/scripts/benchmark_a2000_workstation.py"),
            ("scripts/prolonged_stress_test.py", f"{REMOTE_DIR}/scripts/prolonged_stress_test.py"),
            ("models/unet_3d.py", f"{REMOTE_DIR}/models/unet_3d.py"),
            ("models/diffusion.py", f"{REMOTE_DIR}/models/diffusion.py"),
            ("utils/metrics.py", f"{REMOTE_DIR}/utils/metrics.py"),
        ]
        
        for local_rel, remote_path in files_to_sync:
            local_full = os.path.join(local_base, local_rel)
            if os.path.exists(local_full):
                print(f"Syncing {local_rel} -> {remote_path}...")
                sftp.put(local_full, remote_path)
                
        sftp.close()
        
        # Execute hardware benchmark script on remote workstation
        print("\nExecuting Hardware Benchmark on remote RTX A2000 GPU...")
        cmd = f"cd {REMOTE_DIR} && {py_path} -u scripts/benchmark_a2000_workstation.py"
        stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
        
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line, end="", flush=True)
            
    except Exception as e:
        print(f"Error during remote execution: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
