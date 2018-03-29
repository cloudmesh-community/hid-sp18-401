
## changing password of a headless raspberry pi


 - unplug the raspberry pi cluster and remove the sd card from the slot
 - insert the sd card into a computer, and make sure we should have root 
 access on that computer
 - locate and edit the etc/shadow file on the sd card
 - run the command "openssl passwd -1 -salt"
 - we must find the line that starts with pi and replace the text
   between the first and second with the output from the above command 
   we had executed
 - eject the sd card from the computer
 - insert the sd card back into the pi cluster
 - boot the raspberry pi
 - ssh to log into raspberry pi and set up the password
