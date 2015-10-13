# Notes

These notes are for my personal use. These are not meant to be used by other users - they're just specific notes to myself about which environments, etc. I'm using.

In project interpreter, set: "2.7.10 (~/anaconda/envs/neural-network/bin/python2.7)"

See conda env --help

## Process for recreating the same environment:
- Create environment if it doesn't exist
conda create -n neural-network python

- Switch to the env of choice
source activate neural-network

- Export environment
conda env export > environment.yml

- On the other computer, create the environment if it doesn't exist there
- Must be in the same directory as the environment.yml file
conda env create

- If the environment exists, simply update it
conda env update -f environment.yml

- Activate this new environment:
source activate neural-network

