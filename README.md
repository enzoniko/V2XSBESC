# Instruction to run the code
- Have a .tar.xz with the simulation csv data.
- Have a .sca file with information on the start and end of each vehicle in the simulation.
- Put them into the Data folder.
- Run getVehicleStartEnd.py to get a json file with the start and end of each vehicle.
- Run the generate_data.py to generate the windows10s.hdf5 file with the raw windows for each vehicle.
- Script use_data_test.py contains an example code of how to load the hdf5 to a dictionary and use it. 
    - The script also shows how to separate X and Y and summarize the windows into duration on each mode (steps that will be used in the preprocessing).

# Requirements
First, use a venv to install the requirements, we are are not savages.
```bash
python -m venv venv
source venv/bin/activate
```
You need to have pandas, numpy and h5py installed for now (psutil is optional, but it is used to show the memory usage of the script).:
```bash
pip install -r requirements.txt
```

# Tasks
- Y windows can have less that 10 seconds if they are the last ones. Solve this by removing them?.
    - If we remove this windows directly in generating_data, then we might solve all the problems.
- Investigate why are there so many windows with less than 10 seconds (33633 from 274481) appearing in X and Y. Can we simply remove them?
- Once X and Y windows are correctly generated, we could do a quick EDA (box plots for each mode, for example) to understand the data better.

# Next steps
- Create a pipeline to generate the windows, preprocess them, train, evaluate, and plot many models (Enzo will reuse the codes from the other projects).
- After that, we can start the preprocessing, here we need to decide if we scale the 3 features (duration in each mode) together or separately (idle is way bigger than the others).
- After the preprocessing, another EDA is interesting to understand what changed, if anything changed at all.
- Investigation of models that can handle the format of the problem.
- Implementation of the models and evaluation.
- Grid search for hyperparameters.
- Comparison of the models.
