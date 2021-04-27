#!/bin/bash

idx=0
read -n 1 -s -r -p "Press enter key to start"

while [ $? -eq 0 ]
do
	    echo 'Sleep for 3 sec'
	    sleep 3
        echo 'Prise de vue!'
    	raspistill -w 1920 -h 1080 -o $idx.jpg
    	echo 'Done'
    	idx=`expr $idx + 1`
    	read -n 1 -s -r -p "Press enter key to continue ( ctrl+c to quit )"
done