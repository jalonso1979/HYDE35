import subprocess
try:
    result = subprocess.run(['pre_commit_instructions'], capture_output=True, text=True, check=True)
    print(result.stdout)
except FileNotFoundError:
    print("pre_commit_instructions not found")
except subprocess.CalledProcessError as e:
    print(e.stderr)
