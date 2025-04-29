# Adverse Weather and Road Scenarios Benchmark for Autonomous Driving (AdWeRSBAD)
Public Repository accompanying the SSCI25 paper:
Adverse Weather Benchmarks and Dataset for Object Detection in Autonomous Driving
Authors: Dominik Weikert, Adrian Köring, Christoph Steup

## General Setup:

1. The Code provided in this repository requires a PostgreSQL database backend: https://www.postgresql.org/. You can also use a Docker to deploy Postgresql: https://hub.docker.com/_/postgres. A simple compose.yaml and configuration files can be found in the  dbestup folder of this repository.
2. Copy database.ini to mydatabase.ini and adapt it to your needs. (e.g. fill in your postgresql username and set the project root as well as raw datasets root path correctly). The provided .ini file also containssome examples of database connection parameters.
3. Install this package `pip3 install -e .` (Preferably into a virtual environment using a venv management tool of your choice). A PyPi Version of this package will be uploaded after further testing.
4. To actually use th AdWeRSBAD functionality, you need to obtain the datasets separately:
 - Waymo: [https://waymo.com/open/](https://waymo.com/open/) (Use the v1 version of the dataset)
 - Nuscenes/NuImages: [https://www.nuscenes.org/](https://www.nuscenes.org/)
 - CADCD: [http://cadcd.uwaterloo.ca/](http://cadcd.uwaterloo.ca/)

Extract the datasets into the folder you determined as your dataset root in step 2.

You should now be able to import the datasets into the database using the scripts in dataset_imports/. You can also create the expected table setup in the DB using the provided createtable.py file.


## Dataset Usage:

Once installed, using the dataset is simple via the Adwersbad class (in adwersbad/adwersbad.py).
The class handles the connection to the postgres database, and specific data can be requested via the data parameter, which is expected do be a dictionary, with its keys specifying the table you want to access and the values being a list of columns from that table, eg:

`data={'weather': ['weather'], 'lidar':['points'], camera: ['image']}`

will yield tuples containing the weather annotation, lidar pointclouds and camera images of a sample.

Further specification of the requested data can be made using the other parameters of the dataset:

```
            data List{str} -- database columns to load from the tables
            splits List{str} -- dataset splits {training, validation, testing or all} (default: ['all'])
            scenario {str} -- scenario {e.g. all, rain, night, nightrain or duskdawn} to load (default: {'all'})
            dbtype {str} -- database to load from (default: {'psycopg3@local'})
            itersize {int} -- number of rows to load fom the db at once (default: {100})
            offset {int} -- number of rows to skip before returning data from the db
            limit {int} -- maximum number of rows to return from the db, 0 for no limit. 
            orderby {str} -- column to order the results by, should be a uniquely identifiable column, such as time or a uid
            location {str} -- location to restrict results to
```

After construction, the dataset can then used with a pytorch Dataloader (or simply iterated through) to get all data conforming to the stated specifications. Some usage examples can be found in the test scripts located in test/.



## Weather Labeling Tool

This repository includes a weather labeling tool for manually creating weather labels. Instructions for this will be added to the wiki following the SSCI25 conference.


## References

Waymo Open Dataset: [https://waymo.com/open/](https://waymo.com/open/)

Nuscenes: [https://www.nuscenes.org/](https://www.nuscenes.org/)

Canadian Adwerse Driving Conditions Dataset: [http://cadcd.uwaterloo.ca/](http://cadcd.uwaterloo.ca/)

Open-meteo Weather Database: [https://open-meteo.com/](https://open-meteo.com/)
