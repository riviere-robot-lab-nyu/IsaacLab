# Install leisaac as dependency 

## Enter the same directory as pyproject.toml
```
cd ~/workspace/IsaacLab/leisaac/source
```
## Install editable
```
pip install -e .
```
## Test installation 
```
python -c "import leisaac; print(leisaac.__file__)"
```