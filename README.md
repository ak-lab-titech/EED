## Installation Instructions
- Follow the same installation steps for `habitat-sim` and `habitat-lab` as described in [MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation](https://github.com/saimwani/multiON?tab=readme-ov-file).
- Make sure to install older versions of both `habitat-sim` and `habitat-lab`.

1. Install anaconda
2. Create python3.9 environment:
```
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```
3. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim)
```
cd ..
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
git checkout ae6ba1cdc772f7a5dedd31cbf9a5b77f6de3ff0f
pip install -r requirements.txt; 
python setup.py install --headless # (for headless machines with GPU)
python setup.py install # (for machines with display attached)
```
4. Install [habitat-lab](https://github.com/facebookresearch/habitat-lab)
```
cd ..
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout 676e593b953e2f0530f307bc17b6de66cff2e867
pip install -e .
```
5. Install [LLaVA](https://github.com/haotian-liu/LLaVA)
```
cd ../Main
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```
7. Install [TranSalNet](https://github.com/LJOVO/TranSalNet)
```
cd ..
git clone https://github.com/LJOVO/TranSalNet.git
```
8. Download the pretrained models into the `TranSalNet/pretrained_models` directory
  - [TranSalNet_Dense](https://drive.google.com/file/d/1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T/view): Pretrained on the SALICON training set
  - [DenseNet-161](https://drive.google.com/file/d/1IZ8EtoM7Ui8QA_MlX7lqcIhusLa3ddl6/view): DenseNet-161 pretrained on ImageNet (used for TranSalNet_Dense)
9. Download required data and pretrained models from Google Drive and unzip them.
  - [reconstruction_model](https://drive.google.com/file/d/1RDmKMYATaYM56dpRNHXTNXXBKcmr4dXb/view?usp=drive_link)
  - [models.zip](https://drive.google.com/file/d/1k3uypjxLxqWB2uT53nC_3aC981nAwzBf/view?usp=drive_link)
  - [data.zip](https://drive.google.com/file/d/1gbmcyetXEM_DpsOYkXeis-PoxmccR6ip/view?usp=drive_link)
10. Install the required packages
```
cd ..
pip install -r requirements.txt
```

## Directory Structure
```
├──habitat-sim
├──habitat-lab
├──Main
    ├── README.md
    ├── cpt
    ├── data
    ├── docs
    ├── habitat
    ├── habitat_baselines
    ├── LLaVA
    ├── TranSalNet
    ├── reconstruction_model
    ├── run.py
    │
```

## Usage
- When using the humanoid setting, use `run_humanoid.py` instead of `run.py`.
### Training
- Trained models are saved in the `cpt` directory, with each run saved under a subdirectory named with the corresponding timestamp.
#### EnvDescriber
```
python run.py --run-type train --area-reward-type coverage
```
#### Novelty
```
python run.py --run-type train --area-reward-type novelty
```
#### Coverage
```
python run.py --run-type train --area-reward-type coverage
```
#### Smooth-Coverage
```
python run.py --run-type train --area-reward-type smooth-coverage
```
#### Curiosity
```
python run.py --run-type train --area-reward-type curiosity
```
#### Reconstruction
```
python run.py --run-type train --area-reward-type reconstruction
```

### Evaluation
- The path to the trained model stored in the `cpt` directory should be specified in `run.py`.
#### EnvDescriber
```
python run.py --run-type eval --area-reward-type coverage
```
#### Random
```
python run.py --run-type random
```
#### Novelty
```
python run.py --run-type eval --area-reward-type novelty
```
#### Coverage
```
python run.py --run-type eval --area-reward-type coverage
```
#### Smooth-Coverage
```
python run.py --run-type eval --area-reward-type smooth-coverage
```
#### Curiosity
```
python run.py --run-type eval --area-reward-type curiosity
```
#### Reconstruction
```
python run.py --run-type eval --area-reward-type reconstruction
```
