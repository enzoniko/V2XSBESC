# Instruction to run the code
- Have a .tar.xz with the simulation csv data.
- Have a .sca file with information on the start and end of each vehicle in the simulation.
- Put them into the Data folder.
- Run getVehicleStartEnd.py to get a json file with the start and end of each vehicle.
- Run the generate_data.py to generate the windows10s_summarized.hdf5 file with the summarized windows for each vehicle (and files for all the other generation steps, including rawer forms of data), and to generate X and Y for the models using a configurable number of past windows.


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

# Files
- Data/getVehicleStartEnd.py: Script to get the start and end of each vehicle in the simulation.
- Data/generate_data.py: Script to generate the windows10s_summarized.hdf5 file with the summarized windows for each vehicle.
- preprocessing.py: Script with the functions to preprocess the data.
- models.py: Script with the models.
- ploting.py: Script with the functions to plot the results.
- experiments.py: Script with code that runs models several times and saves the results.
- scratchfile.py: Script with code that is used to test new ideas.


# Next steps
- Sequence models.
- Complex models.
