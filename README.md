# Shift Plotter

This is a tool for plotting signals identfied by Dr. Roger's teams's code. (`correct_accel_gyro_hmm_06_05.py`)
When a signal is identified, a human must use this tool and decide whether the proposed correction is acceptable.
If it is, the user will annotate the region as a real shift and the algorthmic repair is acceptable.
If not, the user will annotate the region as a false positive and the the original signal is acceptable.


# Installation

Clone the repo:

```bash
git clone git@github.com:4d30/shift_plotter.git
cd shift_plotter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

Copy `config.ini.template` to `config.ini` and edit it as needed.
```bash
cp config.ini.template config.ini
vim config.ini
```

Run the script:
```bash
python shift_plotter.py 
```
