#!/bin/bash

# This script authenticates with gcloud and sends the credential file to a remote machine.

# --- Configuration ---
# Please update these values if they are different for your setup.
CREDENTIAL_PATH="/home/sattoutah/.config/gcloud/application_default_credentials.json"
REMOTE_USER="statge"  # <-- IMPORTANT: Change this to the username on the remote machine
REMOTE_HOST="192.168.2.169"
REMOTE_PATH="/home/$REMOTE_USER/.config/gcloud/" # Assumes same structure

# --- Step 1: gcloud Authentication ---
echo "Step 1: Authenticating with gcloud..."
# This will open a browser for you to log in.
gcloud auth application-default login
if [ $? -ne 0 ]; then
    echo "❌ gcloud authentication failed. Please check your gcloud installation and configuration."
    exit 1
fi
echo "✅ gcloud authentication successful."

# --- Step 2: Check for credential file ---
if [ ! -f "$CREDENTIAL_PATH" ]; then
    echo "❌ Error: The gcloud credential file does not exist at: $CREDENTIAL_PATH"
    echo "Please ensure the authentication step was successful."
    exit 1
fi
echo "✅ Credential file found."

# --- Step 3: Create remote directory ---
echo -e "\nStep 3: Creating directory on remote machine..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_PATH}"
if [ $? -ne 0 ]; then
    echo "❌ Failed to create directory on remote machine. Please check your SSH connection and permissions."
    exit 1
fi
echo "✅ Remote directory ensured."

# --- Step 4: Copy credentials ---
echo -e "\nStep 4: Copying credentials to remote machine..."
echo "Attempting to copy '$CREDENTIAL_PATH' to '${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}'..."

scp "$CREDENTIAL_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

if [ $? -eq 0 ]; then
    echo -e "\n✅ File copied successfully."
else
    echo "❌ SCP command failed. Please check your network connection, credentials, and SCP path."
    exit 1
fi

echo -e "\n✅ Done."