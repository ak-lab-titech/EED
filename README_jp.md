## インストール方法
- habitat-simとhabitat-labのインストールまでは「[MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation](https://github.com/saimwani/multiON?tab=readme-ov-file)」と同じ
- habitat-simやhabitat-labは古いバージョンをインストールする

1. anacondaをインストール
2. python3.9環境を作成:
```
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```
3. [habitat-sim](https://github.com/facebookresearch/habitat-sim)をインストール:
```
cd ..
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim 
git checkout ae6ba1cdc772f7a5dedd31cbf9a5b77f6de3ff0f
pip install -r requirements.txt; 
python setup.py install --headless # (for headless machines with GPU)
python setup.py install # (for machines with display attached)
```
4. [habitat-lab](https://github.com/facebookresearch/habitat-lab)をインストール
```
cd ..
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout 676e593b953e2f0530f307bc17b6de66cff2e867
pip install -e .
```
5. [LLaVA](https://github.com/haotian-liu/LLaVA)をインストール
```
cd ../Main
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
```
7. [TranSalNet](https://github.com/LJOVO/TranSalNet)をインストール
```
cd ..
git clone https://github.com/LJOVO/TranSalNet.git
```
8. `TranSalNet/pretrained_models`ディレクトリに事前学習済みモデルをダウンロードする
  - SALICON トレーニング セットで事前トレーニングされたモデル([TranSalNet_Dense](https://drive.google.com/file/d/1JVTYq5UE6Q0OHoOVoXWF5WW5w42jlM1T/view))
  - ImageNet で DenseNet-161 (TranSalNet_Dense) の事前トレーニング済みモデル([DenseNet-161](https://drive.google.com/file/d/1IZ8EtoM7Ui8QA_MlX7lqcIhusLa3ddl6/view))
9. 必要なデータや学習済みモデルを研究室サーバーからコピーする
  - `/mnt/nasA/matsumoto/data`をコピー
  - `/mnt/nasA/matsumoto/reconstruction_model`をコピー
  - `/mnt/nasA/matsumoto/models`をcptとしてコピー
10. 必要なパッケージをインストールする
```
cd ..
pip install -r requirements.txt
```

## ディレクトリ構成
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

## 使い方
- humanoidありの場合は、`run.py`ではなく`run_humanoid.py`を使う
### 学習
- 学習したモデルはcptディレクトリに実行時の日時をディレクトリ名として保存される
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

### 評価
- cptディレクトリに格納されている学習済みモデルのパスをrun.pyで指定する
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
