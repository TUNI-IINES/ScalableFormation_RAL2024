# CCTA2024_ScalableFormation

This is the repository for the application used for the paper submission toward CCTA2024.

The main authors are Hoang Pham, Nadun Ranasinghe, and Dong Le.

The repository was inherited from and supported by Widhi Atman's repository.


## Brief Description
There are three branches:
- `main`: the stable program that has been tested.
- `algorithm_testing`: the unstable one for implementing new theories.
- `experiment`: the ROS environment needs to pull the `main` or `algorithm_testing` branches to the `\src` folder.


## Instructions
To run Python simulations, simply run `python sim2D_main.py` on the root directory.

To run ROS simulations, run `python expROS_main.py` on the root directory.\
After the run, the application will generate a folder in `animation_results` with the `.pkl` file as the recording and some plots.

To run ROS experiments, please ask Nebolab administrators for further instructions.\
To modify the scenario, adjust the parameters in the `scenarios_unicycle\scenarios\*.yml` file.\
Ensure that the name of the `.yml` file should be the same as the one mentioned in `scenarios_unicycle\CCTA2024_FormationObstacleLidar_scenario.py`.

To plot `.pkl` file, put the `<name>.pkl` in `scenarios_unicycle\saved_pkl\`.\
Uncomment `exp_pkl_plot()` in `pickleplot.py`, then at root directory, run `python pickleplot.py <name>`.

To process the video, put both `<name>.mp4` and `<name>.pkl` in `scenarios_unicycle\saved_pkl\`.
First, uncomment `exp_video_pkl_plot(beautify=True)` in `pickleplot.py`, then at root directory, run `python pickleplot.py <name>`.
Second, uncomment `exp_video_pkl_plot(snap=True)` in `pickleplot.py`, then at root directory, run `python pickleplot.py <name>` again.
