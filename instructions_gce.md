# Instructions for running GCE

## Overview
We perform computation heavy tasks within the google cloud engine framework.

## Setup Instructions

### 1. Connecting to the VM Instance

To connect to the VM Instance, start the VM and then click on it. A button "SSH" with a dropdown will show up.
Click on the button and allow connection with SSH keys (automatically generated - so don't need to worry about anything)

You will see a command line and youre credentials logged in.
```
<name>@adlr-trainingspipeline:~$
```
Connect to a tmux session so training / SSH credentials are not interrupted due to time out
```
tmux new -s <session_name>
```
Now navigate to the local repository by typing
```
cd tum-adlr-wise24-17
```
You are now in the working repository.

### 2. Setup PAT

To have a clean working schedule and not any accidential merging issues, we have to ensure that we have pulled the latest commits from the origin main. To be able to do so, we need to initialize a personal access token (PAT) from github, as password logins are not supported. To do so, go into your **Github Account>Settings>Developer Settings>Personal acces tokens>Tokens (classic)**. There, set up a new token and name it properly (e.g. TUM ADLR GCE Instance). Github will generate this token only once, so make sure you save it (I saved it in a word file with a command cheat sheet). It should look something like this:

```
ghp_kUHp5DTaseE5MSCfK6Skp8HegNnvKLsTu89 (not a real token)
```
Use this PAT when prompted for your password within the VM.

### 2. Git Push/Pull

Now we can use our github username and our PAT to pull the latest commits into our local repository

```bash
git pull origin main
Username: <username>
Password: <PAT>
```
When this is done, we can work in the repository just as we would on our local machine. To run scripts, simply call the python interpreter and choose the script which you want to run.

```bash
python3 scripts/<example.py>
```

When finished on the VM, stage, commit and push all files just as you would on your local machine. Keep in mind that we don't want to upload any models onto the online repo, therefore include them in the `.gitignore`.

```bash
git add <filename> or .
git commit -m "insert commit log"
git push origin <branch name> (normally main)
```

You will then be prompted to enter your credentials

```bash
Username: <username>
Password: <PAT>
```

When finished in the VM, end the tmux session
```bash
tmux kill-session -t session_name
```

When all changes have been commited, you can stop the VM instance in the GCE interface.
