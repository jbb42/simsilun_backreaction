#!/bin/bash
# Install dependencies for S-GenIC on Ubuntu/Debian
# chmod +x SGenIC_Ubuntu.sh

set -e  # exit if any command fails

echo "Updating package database..."
sudo apt update -y
sudo apt upgrade -y

echo "Installing development tools..."
sudo apt install build-essential g++ make cmake -y

echo "Installing GSL..."
sudo apt install libgsl-dev -y

echo "Installing FFTW..."
sudo apt install libfftw3-dev -y

echo "Installing HDF5..."
sudo apt install libhdf5-dev -y

echo "Installing optional but useful tools..."
sudo apt install libtool automake autoconf -y

echo "All dependencies installed!"
