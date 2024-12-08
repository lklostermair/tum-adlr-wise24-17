# Instructions for Running GCE

## Overview
We perform computation-heavy tasks within the Google Cloud Engine (GCE) framework. These instructions guide you through connecting to the VM, setting up your environment, and using Git effectively.

---

## Setup Instructions

### 1. Connecting to the VM Instance
1. Start the VM Instance:
   - Go to the GCE interface, locate your VM, and start it.
   
2. Connect via SSH:
   - Click on the VM, then click the SSH button. Allow connection with the automatically generated SSH keys.

3. Access the Command Line:
   - You should see the command line with your credentials logged in:
     ```
     <name>@adlr-trainingspipeline:~$
     ```

4. Start a tmux Session:
   - To prevent disconnection due to timeouts, connect to a `tmux` session:
     ```
     tmux new -s <session_name>
     ```

5. Navigate to the Local Repository:
   - Enter the working directory:
     ```
     cd tum-adlr-wise24-17
     ```

---

### 2. Setup Personal Access Token (PAT)

To manage Git operations securely and avoid merge issues:
1. Generate a PAT:
   - In your GitHub account, go to:
     `Settings > Developer Settings > Personal Access Tokens > Tokens (classic)`.
   - Generate a new token and name it appropriately (e.g., "TUM ADLR GCE Instance").
   - Save the token securely. GitHub displays it only once. Example:
     ```
     ghp_kUHp5DTaseE5MSCfK6Skp8HegNnvKLsTu89 (not a real token)
     ```

2. Use the PAT for Authentication:
   - When prompted for a password in the VM, use your GitHub username and the PAT.

---

### 3. Git Push/Pull

1. Pull the Latest Commits:
   - Ensure the local repository is up-to-date:
     ```
     git pull origin main
     Username: <username>
     Password: <PAT>
     ```

2. Running Scripts:
   - Use the Python interpreter to execute scripts:
     ```
     python3 scripts/<example.py>
     ```

3. Stage, Commit, and Push Changes:
   - Ensure no models are pushed (add them to `.gitignore`):
     ```
     git add <filename> or .
     git commit -m "insert commit log"
     git push origin <branch name> (normally main)
     ```
   - Enter credentials when prompted:
     ```
     Username: <username>
     Password: <PAT>
     ```

---

### 4. Ending the Session

1. End the tmux Session:
   - When finished, terminate the session:
     ```
     tmux kill-session -t <session_name>
     ```

2. Stop the VM:
   - Stop the VM instance in the GCE interface.

---
