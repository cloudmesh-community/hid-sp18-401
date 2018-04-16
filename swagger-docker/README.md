Assignment: Cloud and Big Data Rest Service with Swagger

Service Description
 Calculate average  mean square error for an ‘advertisement dataset’ (from the webpage : http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv) after performing 10 fold cross validation on the Linear Regression model 

Here, I have implemented a cv object that will return cv mse error. The swagger code-gen generate the server stub code for us by taking the cv.yaml as input and gives us a foundation to develop the main logic. 

Main logic : cv_stub.py 
Location - https://github.com/cloudmesh-community/hid-sp18-401/swagger-docker /log_stub.py

You can download the code from the repository and test or enhance further.

Follow the below steps
Clone the repository
Makefile and Dockerfile are also in the above location

After you run sudo make docker-all command from your terminal
Docker will create an Ubuntu image, and after docker image is created, make file executes series of commands, creating my swagger REST service in the docker container
 
You will see something like this Running on http://0.0.0.0:8080/ (Press CTRL+C to quit)

Running the service from browser
Open any browser and request with DB_URL Ex: Input : http://0.0.0.0:8080/api/db 

Output : {
  "mse": "3.059967618118514"
}
Here mse means mean squared error  
