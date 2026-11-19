import os
os.environ["NETWORK_PROFILE"] = "grid3x3_6tl"

import sys
sys.path.insert(0, "src")

import traci
from env import _sumo_cmd
import config

traci.start(_sumo_cmd(config.ROUTE_FILES["peak"]))

tls_ids = [
    t for t in traci.trafficlight.getIDList()
    if len(traci.trafficlight.getAllProgramLogics(t)[0].phases) > 1
]

print("Trained junctions:", sorted(tls_ids))
print("Count:", len(tls_ids))

traci.close()