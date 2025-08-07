# Edge Code Reactions Library

Repository for Particle Tracker Reactions library, based on NESO-Particles.

## Quick-start with docker
Ensure docker v28.1.1+ is installed on either linux or WSL2 with non-root management enabled. 
See here for more details: <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>.

Clone the repo:
```
git clone --recurse-submodules git@github.com:UKAEA-Edge-Code/Reactions.git $HOME/VANTAGE_Reactions
```

Feel free to replace ``$HOME/VANTAGE_Reactions`` with a directory name of your choice.
Next from within the repo directory, execute:
```
docker build -t vantage_reactions_img -f .devcontainer/Dockerfile .
```
then execute:
```
docker run --name vantage_reactions -v "$(pwd):/root/Reactions" vantage_reactions_img:latest -c "cd /root/Reactions && source run_tests.sh"
```

## Documentation
To build the documentation ensure the following pre-requisites are installed (via `pip`):

- sphinx (https://pypi.org/project/Sphinx/)
- pydata-sphinx-theme (https://pypi.org/project/pydata-sphinx-theme/)
- breathe (https://pypi.org/project/breathe/)

This can be done manually or by running:
```
pip install -U -r $HOME/VANTAGE_Reactions/docs/sphinx/requirements.txt
```
(replace `$HOME/VANTAGE_Reactions` with the name of the directory that the repo was cloned into.)

Then simply run `make` inside the `docs` folder in the repo. The documentation should be contained within:
```
$HOME/VANTAGE_Reactions/docs/build/sphinx/html/
```
Start by opening `index.html` in a web browser of your choice.