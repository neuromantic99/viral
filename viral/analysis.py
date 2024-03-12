from pathlib import Path
from matplotlib import pyplot as plt
from models import TrialInfo


data_path = Path("data")
session_number = "023"

session_path = data_path / session_number

n_trials = len(list(Path(session_path).glob("trial*.json")))

for trial in range(n_trials):
    with open(session_path / f"trial{trial}.json") as f:
        data = TrialInfo.model_validate_json(f.read())

    print(data.licks)
    plt.plot(data.rotary_encoder_position)
plt.show()
