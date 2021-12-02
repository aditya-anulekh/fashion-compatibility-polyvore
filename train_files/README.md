# Training and Validation files

Each of the files in this directory is generated using the ```compatibility_*.txt``` file and ```train.json``` or ```valid.json``` in order to avoid fetching file paths every time the DataLoader/Dataset is called. 

We assume that all pairs of images listed in a row in the ```compatibility_*.txt``` file are either compatible with each other or not.