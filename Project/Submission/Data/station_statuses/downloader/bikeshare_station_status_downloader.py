import json
import requests
from datetime import datetime

url = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_status"
response = requests.get(url)
filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S") + " station_status.json"
open(filename, "wb").write(response.content)
print("Downloaded " + filename)
