# Notes

These notes are for my personal use. These are not meant to be used by other users - they're just specific notes to myself about which environments, etc. I'm using.

In project interpreter, set: "2.7.10 (~/anaconda/envs/neural-network/bin/python)"

See conda env --help

## Process for recreating the same environment:

- Switch to the env of choice
source activate neural-network

- Export list of packages
conda list --explicit > packages.txt

- On the other computer, remove the env if it exists
conda env remove -n neural-network

- Verify removal
conda info --envs

- Make new environment with all packages:
conda create -n neural-network --file packages.txt

- Activate this new environment:
source activate neural-network