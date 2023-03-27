# MLOps_POC

A simple classification problem with sensor data. It classifies if devices are faulty or not. 
The focus of this project is not to deal with complex data or models but to mainly prepare production ready code and deploy them in a cloud architecture.

# How to run this project?

Prerequirements:

1. Pip
2. Python 3.10.10

In your shell terminal run the following commands:

python -m pip install --upgrade pip
pip install -r requirements.txt

Git clone the repo
Git checkout 'develop' branch. 
Run the main.py file of the project either in any IDE or terminal.

Note: the project contains .github/workflow/ci.yml to enable github actions. Your CI-CD pipeline will run automatically once you push any changes to the repository. 
