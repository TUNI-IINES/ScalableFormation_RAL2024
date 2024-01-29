# CCTA2024_ScalableFormation
This is the repository for the application used in the course KONE.533 Robotics Project Works and for the paper submission toward CCTA2024.
The main authors are: Hoang Pham, Nadun Ranasinghe, and Dong Le.
The repository were inherited from the Widhi Atman's repository.

## Brief Description
There are three branches:
- `main`: the stable program that has been tested.
- `algorithm_testing`: the unstable one, for implementing new theories.
- `experiment`: the ROS environment, needs to pull the `main` or `algorithm_testing` branches to the `\src` folder

## Instructions
To run Python simulations, simply run `python sim2D_main.py` on the root directory.

To run ROS simulations, run `python expROS_main.py` on the root directory.

To run ROS experiments, please ask Nebolab adminstrators for further instructions.
To modify the scenario, adjust the parameters in `scenarios_unicycle\scenarios\*.yml` file. 
Make sure that the name of the `.yml` file should be the same as the one mentioned in `scenarios_unicycle\CCTA2024_FormationObstacleLidar_scenario.py`.

After the run, the application will generate a folder in `animation_results` with `.pkl` file as the recording and some plots.
