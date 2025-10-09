#!/bin/bash
# Install dependencies for S-GenIC on Fedora
# chmod +x SGenIC_Fedora.sh

set -e  # exit if any command fails

echo "Updating package database..."
sudo dnf update -y

echo "Installing development tools..."
sudo dnf groupinstall "Development Tools" -y
sudo dnf install gcc-c++ make cmake -y

echo "Installing GSL..."
sudo dnf install gsl-devel -y

echo "Installing FFTW..."
sudo dnf install fftw-devel -y

echo "Installing HDF5..."
sudo dnf install hdf5-devel -y

echo "Installing optional but useful tools..."
sudo dnf install libtool automake autoconf -y

echo "All dependencies installed!"
