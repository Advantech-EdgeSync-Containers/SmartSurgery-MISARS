#!/bin/bash
# ==========================================================================
# Smartsurgery-Misar Docker Compose Build Script
# ==========================================================================
# Version:      25.24
# Author:       BoAn (boan.tsai@smartsurgerytek.com), ......#kindly append your name here
# Created:      June 18, 2025
# ==========================================================================

clear

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}"
echo " ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗██████╗ ██╗   ██╗"
echo " ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║   ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗╚██╗ ██╔╝"
echo " ███████╗██╔████╔██║███████║██████╔╝   ██║   ███████╗██║   ██║██████╔╝██║  ███╗█████╗  ██████╔╝ ╚████╔╝ "
echo " ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║   ╚════██║██║   ██║██╔══██╗██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝  "
echo " ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║   ███████║╚██████╔╝██║  ██║╚██████╔╝███████╗██║  ██║   ██║   "
echo " ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   "
echo -e "${WHITE}                                  Center of Excellence${NC}"
echo
echo -e "${CYAN}  Proceeding...${NC}"
echo

sleep 7

mkdir -p src models data diagnostics

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Checking X environment variables..."
echo "XAUTHORITY=$XAUTHORITY"
echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"

if [ -z "$XAUTHORITY" ] || [ -z "$XDG_RUNTIME_DIR" ]; then
    echo "Setting up X11 forwarding..."
    if [ -z "$XAUTHORITY" ]; then
        XAUTH_PATH=$(xauth info 2>/dev/null | grep "Authority file" | awk '{print $3}')
        if [ -n "$XAUTH_PATH" ]; then
            export XAUTHORITY=$XAUTH_PATH
            echo "XAUTHORITY set to $XAUTHORITY"
        fi
    fi
    if [ -z "$XDG_RUNTIME_DIR" ]; then
        export XDG_RUNTIME_DIR=/run/user/$(id -u)
        echo "XDG_RUNTIME_DIR set to $XDG_RUNTIME_DIR"
    fi
    if command_exists xhost; then
        echo "Configuring xhost access..."
        xhost +local:docker
        touch /tmp/.docker.xauth
        xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -
        chmod 777 /tmp/.docker.xauth
    else
        echo "Warning: xhost command not found. X11 forwarding may not work properly."
    fi
else
    echo "X environment variables already set, skipping X11 setup."
fi

echo "Starting Docker containers..."
if command_exists docker-compose; then
    echo "Using docker-compose command..."
    docker-compose up -d
elif command_exists docker && command_exists compose; then
    echo "Using docker compose command..."
    docker compose up -d
else
    echo "Error: Neither docker-compose nor docker compose commands are available."
    exit 1
fi

echo "Connecting to container..."
docker exec -it smartsurgery-misar bash