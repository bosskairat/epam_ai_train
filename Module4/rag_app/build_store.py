import os
import platform
import subprocess
from dotenv import load_dotenv


# --- VECTOR DATABASE CONFIGURATION ---
WEAVIATE_CONTAINER_NAME = "simple-rag-weaviate"
WEAVIATE_IMAGE = "semitechnologies/weaviate:1.33.7"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051
load_dotenv()
WEAVIATE_HTTP_PORT_EXTERNAL = os.environ["WEAVIATE_HTTP_PORT_EXTERNAL"]
WEAVIATE_GRPC_PORT_EXTERNAL = os.environ["WEAVIATE_GRPC_PORT_EXTERNAL"]

# --- WSL Detection ---
system = platform.system()
USE_WSL = system == "Windows"
print(f"Operating System: {system}. Using WSL for Docker commands: {USE_WSL}")

# --- Shell Command Helpers ---
def run_wsl_command(command):
    """Executes a command inside WSL and returns the result."""
    result = subprocess.run(
        ["wsl", "-e", "bash", "-l", "-c", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0
    }

def run_linux_command(command):
    """Executes a command in a standard Linux/macOS shell."""
    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0
    }

def run_shell_command(command):
    """Universal function to run a shell command, abstracting WSL usage."""
    if USE_WSL:
        return run_wsl_command(command)
    else:
        return run_linux_command(command)


# First, ensure no old container with the same name is running
print(f"--- Stopping and removing any existing container named '{WEAVIATE_CONTAINER_NAME}' ---")
stop_command = f"docker stop {WEAVIATE_CONTAINER_NAME} 2>/dev/null; docker rm {WEAVIATE_CONTAINER_NAME} 2>/dev/null"
res = run_shell_command(stop_command)
print("Cleanup complete.")

# Now, run the new Weaviate container
print(f"\n--- Starting Weaviate container '{WEAVIATE_CONTAINER_NAME}' ---")
run_command = (
    f"docker run -d "
    f"--name {WEAVIATE_CONTAINER_NAME} "
    f"-p {WEAVIATE_HTTP_PORT_EXTERNAL}:{WEAVIATE_HTTP_PORT} "
    f"-p {WEAVIATE_GRPC_PORT_EXTERNAL}:{WEAVIATE_GRPC_PORT} "
    f"-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true "
    f"-e PERSISTENCE_DATA_PATH=/var/lib/weaviate "
    f"-e DEFAULT_VECTORIZER_MODULE=none "
    f"-e ENABLE_MODULES='' "
    f"-e CLUSTER_HOSTNAME=node1 "
    f"{WEAVIATE_IMAGE}"
)

result = run_shell_command(run_command)

if result["success"]:
    print("✅ Weaviate container started successfully.")
    print("Waiting a few seconds for the service to initialize...")
    import time
    time.sleep(10) # Give Weaviate time to start up
else:
    print("❌ Failed to start Weaviate container.")
    print(f"Stderr: {result['stderr']}")

# Display container statistics
print("\n--- Weaviate Container Stats ---")
stats_result = run_shell_command(f"docker stats {WEAVIATE_CONTAINER_NAME} --no-stream")
print(stats_result["stdout"])
if stats_result["stderr"]:
    print(f"Stderr: {stats_result['stderr']}")