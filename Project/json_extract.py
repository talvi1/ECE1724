import os, json
import datetime 
import pandas as pd
import requests

json_files = [x for x in os.listdir('./data/json/')]

d = []
for index, js in enumerate(json_files):
    with open(os.path.join('./data/json/', js)) as json_file:
        data = json.load(json_file)
        stations = data["data"]["stations"]
        for x in range(len(stations)):
            d.append(
                {
                    'Station_ID': stations[x]["station_id"], 
                    'Bikes_Available': stations[x]["num_bikes_available"], 
                    'Bikes_Disabled': stations[x]["num_bikes_disabled"],
                    'Docks_Available': stations[x]["num_docks_available"],
                    'Docks_Disabled': stations[x]["num_docks_disabled"], 
                    'Last_Updated': data["last_updated"]
                }
            )
df = pd.DataFrame(d)

print(df.head(5))
print(df.tail(5))

