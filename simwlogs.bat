@echo off
title Python ODE Simulation - Tumorigenesis and axon regulation for pancreatic cancer
color 06


python tumor.py > logs.txt

type logs.txt
pause
