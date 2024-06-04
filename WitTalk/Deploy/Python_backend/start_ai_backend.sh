#!/bin/bash

if [ -e "preprocessing"]; then 
	cd AI_SERVER
	python AIdog_server.py


else
	
	cd AI_SERVER
	../preprocessing/bin/python AIdog_server.py

fi