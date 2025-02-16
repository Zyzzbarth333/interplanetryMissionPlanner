# README.md
# Interplanetary Mission Planner

## Project Setup

### SPICE Kernel Setup
This project uses SPICE kernels from NASA's NAIF service for planetary ephemeris calculations. Due to their size, these kernels are not included in the repository and need to be downloaded separately.

#### Required Kernels
Based on your project structure, you need the following kernels:

1. LSK (Leap Seconds Kernel):
   - naif0012.tls

2. PCK (Planetary Constants Kernel):
   - earth_620120_240827.bpc
   - gm_de440.tpc
   - mars_iau2000_v1.tpc
   - moon_pa_de440_200625.bpc
   - pck00011.tpc

3. SPK (Spacecraft and Planet Kernel):
   - Comets: tson.bsp, siding_spring_8-19-14.bsp, siding_spring_s46.bsp
   - Jupiter: jup344.bsp, jup344-s2003_j24.bsp, jup346.bsp, jup365.bsp
   - Mars: mar097.bsp
   - Neptune: [list of your Neptune kernels]
   - Pluto: plu060.bsp
   - Saturn: [list of your Saturn kernels]
   - Uranus: [list of your Uranus kernels]

#### Downloading Kernels
1. Visit the NASA NAIF website: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
2. Navigate to the appropriate directories for each kernel type:
   - LSK: /lsk
   - PCK: /pck
   - SPK: /spk

3. Download the required kernels and place them in the corresponding directories under `data/ephemeris/spice/kernels/`

Alternatively, you can use the provided `load_kernels.py` script to automatically download the required kernels:

```bash
python load_kernels.py
```

### Project Structure
```
interplanetryMissionPlanner/
├── data/
│   └── ephemeris/
│       └── spice/
│           ├── kernels/
│           │   ├── lsk/
│           │   ├── pck/
│           │   └── spk/
│           └── meta/
├── src/
└── README.md
```

### Installation
1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SPICE kernels as described above

### Usage
[Add usage instructions specific to your project]

## Contributing
[Add contribution guidelines if needed]

## License
[Add your license information]