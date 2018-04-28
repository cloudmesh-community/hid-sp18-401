# Final Project: REST Service Framework for Singular Value Decomposition on MNIST Image Classification Problem


## Service Description

Dockerized Rest-swagger webservice  that trains a fully connected neural network
for  MNIST image classification problem and applies singular value decomposition
according to size of SVD and optimize the network

Here, I have implemented a svd object that will return test accuracies of
compressed network. The swagger code-gen generate the server stub code for us by
taking the svd.yaml as input and gives us a foundation to develop the main
logic.

## Main logic : svd_stub.py

Location - https://github.com/cloudmesh-community/hid-sp18-401/project-code/cloudmesh/svd/svd_stub.py
You can download the code from the repository and test or enhance further.

## Steps to follow 

Follow the below steps Clone the repository Makefile and Dockerfile are also in
the above location

After you run sudo make docker-all command from your terminal Docker will create
an Ubuntu image, and after docker image is created, make file executes series of
commands, creating my swagger REST service in the docker container

You will see something like this Running on http://0.0.0.0:8080/ (Press CTRL+C
to quit)

Running the service from browser Open any browser and request with DB_URL Ex:
Input : http://0.0.0.0:8080/api/svd

Here is the output displayed

  "acc": "The Original MNIST Network test accuracy is 0.9742 After doing SVD on
  "the original network with D value as 10, we get a test accuracy of 0.9514
  "With D value as 20, we get a test accuracy of 0.9551 With D value as 50, we
  "get a test accuracy of 0.9555 With D value as 100, we get a test accuracy of
  "0.8997 With D value as 200, we get a test accuracy of 0.9478"


## Points to note

The entire code including time to create docker image will take around 10 mins,
this is due to heavy and large tensorflow computations involoved in the code.

The accuracies vary from run to run because of the training differences of the
newtork from run to run
