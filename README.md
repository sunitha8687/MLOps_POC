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

# To deploy the machine learning solution as a microservice

1. Install fastAPI
2. Setup app.py to perform inference.

# Unit testing and Integration testing.

1. Install pytest.  
2. Write unit tests for your machine learning modules such as:
    test_preprocess.py
    test_predict.py 
3. Create intgration test as test_app.py to make sure that your integration as a microservice works fine.

# How to check if your application performs predictions from your local ? 

1. Install Postman app.
2. pip install uvicorn 
3. To start the app execute the command from the terminal: uvicorn app:app --reload 
4. Create a post request and edit the body before pressing 'send'. Check the ouput. 

Note: the project contains .github/workflow/cicd.yml to enable github actions. Your CI-CD pipeline will run automatically once you push any changes to the repository. 

# How to setup CI - CD for this project? 

1. Create two folders in the root .github/workflows and then create cicd.yml file. 
2. Define the necessary build, test and deploy jobs in yaml format.
3. Create ECR repository in AWS and connect to ECR using github actions. 
4. Create Dockerfile
5. Deploy job uses AWS cloud as a platform to deploy the application as docker image in ECR repository. 


