## Overview
This repository contains a generic Python implementation of a Genetic Algorithm to solve the Travelling Salesman Problem (TSP). 
Geographic coordinates of cities are provided as input to generate a edge-weighted complete graph where the weights are the distance between the cities.
The Adaptive Genetic Algorithm with adaptive crossover rate and mutation rate has ran on berlin52 and presented below.

## Adaptive Genetic Algorithm on berlin52
<div  align="center">    
<img src="https://github.com/lizhaoliu-Lec/TSP-GA/blob/master/results/sample_result.png" width = "600" height = "300" alt="sample_result" />
<img src="https://github.com/lizhaoliu-Lec/TSP-GA/blob/master/results/vis.PNG" width = "600" height = "300" alt="sample_result" />
</div>
    
## Instructions
- Install required packages:
	```bash
	pip install -r requirements.txt
	```
- Good to go!
    ```bash
    python main.py 
    ```
   
## Support Data
- berlin52
- pr76
- rat99
- lin105

## Reference   
- [lccasagrande/TSP-GA](https://github.com/lccasagrande/TSP-GA)
- [ytLab/gaft](https://github.com/PytLab/gaft)

