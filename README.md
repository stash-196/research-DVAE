# This Forked Repository

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/stash-196/research-DVAE.git
cd research-DVAE
```

### 2. Install Python and Set Up Virtual Environment
Ensure you have **Pyenv** installed. Then, create and activate a Pyenv-managed virtual environment:
```sh
pyenv install 3.12.7  # If not already installed
pyenv virtualenv 3.12.7 research-dvae-env
pyenv local research-dvae-env
```

### 3. Install Dependencies
```sh
pip install --upgrade pip
pip install -e .
pip install -r dev-requirements.txt
```

### 4. Directory Structure Overview
```
research-DVAE/
├── setup.py                # Project setup file
├── requirements.txt        # Main dependencies
├── dev-requirements.txt    # Development dependencies
├── src/                    # Source code
│   ├── dvae/               # Main package
│   ├── ...
├── bin/                    # Executable scripts
├── slurm/                  # SLURM job scripts
├── scripts/                # Bash scripts
├── data/                   # Git-ignored datasets
├── outputs/                # Model outputs and logs
```

Now, your environment is set up and ready to use! 🚀

## Deigo train / eval loop

Edit and test train/eval on the **Studio**. Sync code with **git** to Deigo home `~/workspace/research-DVAE`. Then on Deigo:

1. Edit `bin/generate_config_file_for_multiple.py` — experiment name and the sweep lists (dataset, model, ptf, dims, etc.).
2. Load Singularity and generate the batch configs:
   ```sh
   ml singularity
   singularity exec /bucket/DoyaU/stash/containers/generic_ml_container.sif python3 bin/generate_config_file_for_multiple.py
   ```
3. Check that the printed experiment count matches what you meant. Configs land in `config/general_signal/generated/<experiment_name>/`. Generate-config also prints the submit command.
4. Submit that batch (pass the experiment name; do not edit the script just to point at a folder):
   ```sh
   bash scripts/slurm/train/run_training_multiple.sh <experiment_name>
   ```
   That writes one temp `.slurm` per `.ini` and `sbatch`s them. Extra names as extra args if you are submitting more than one batch.
5. `squeue` to confirm they landed, then monitor.

**Outputs** — training writes `/flash/DoyaU/stash/research-DVAE/saved_model`. Copy the date folders you care about to `/bucket/DoyaU/stash/research-DVAE/saved_model` so the Studio mount can see them. Helper: `scripts/bash/sync_saved_models_btw_clusters.sh` (uncomment the date dirs first).

Eval / aggregate after a pull: `scripts/slurm/evaluation/run_eval_multiple.sh`, `scripts/slurm/evaluation/run_aggregate_multiple.sh`. Overlay finished aggregates (OTF vs interpolate vs indicate, …): `scripts/slurm/evaluation/run_compare_aggregates.sh` (optional `DEPENDENCY_JOBIDS=123,456` → `sbatch --dependency=afterok:…`).

Paths for this machine vs Deigo live in `config/device_paths.yaml`. Studio `~/mounts/bucket/...` is a local cache, not a live cluster mount.

## Baselines: GRU-D and SINDy

These sit beside the OTF trainer. They do not change `bin/train_model.py`, `learning_algo.py`, or `bin/generate_config_file_for_multiple.py`. Generated ini files stay under `config/general_signal/generated/` and are gitignored.

### GRU-D

Che et al. 2018. Input decay and hidden-state decay use the missingness mask and the time since the last observation. NaNs are the mask; do not pre-impute, and do not use an `_indicate` or `_interpolate` observation (the cell builds the mask itself).

`Network.name = GRU_D` is not registered in `LearningAlgorithm` (that trainer is in use). Train with the separate entry point. On Deigo, after `git pull`:

```sh
ml singularity
singularity exec /bucket/DoyaU/stash/containers/generic_ml_container.sif python3 bin/generate_gru_d_configs.py
bash scripts/slurm/train/run_gru_d_multiple.sh gru_d_lorenz_baselines
```

Edit the sweep lists in `bin/generate_gru_d_configs.py` (observation, mask label, epochs) before generating. `mask_label` must match a mask pickle already on disk; `None` is fully observed. Set `epochs` beside the OTF runs you want to compare against. A single hand-written example is `config/lorenz63/cfg_gru_d.ini`:

```sh
python bin/train_gru_d.py --cfg config/lorenz63/cfg_gru_d.ini
```

On a host listed in `config/device_paths.yaml` (Deigo), `data_dir` and `saved_root` come from that file. Otherwise the ini paths are used. Checkpoints go under `{saved_root}/gru_d/`, not into an OTF experiment folder. The job writes `evaluation_summary.yaml` with `mse_tf`, `imputation_mse` (Lorenz only, where the unmasked series exists), and the half-series closed-loop scores `mse_auto`, `kld_auto`, `spectrum_error_auto`.

GRU-D is trained teacher-forced on observed samples. The closed-loop numbers use the same `half_half` schedule and metric names as the OTF eval, but the model was not trained with scheduled autonomous steps.

PhysioNet 2012, once that benchmark's parquet is prepared: set `dataset_name = PhysioNet2012`, `observation_process = raw_vitals` (width 5) or `raw_hr` (width 1), `sequence_len = 48`, `delta_step = 1` (one hour per row). `bin/train_gru_d.py` already calls `build_dataloader`, so no loader change is required. Truly missing ICU hours have no imputation label; `imputation_mse` stays empty and `mse_tf` is the error on observed hours.

To run GRU-D inside the live trainer later, add a `GRU_D` branch in `LearningAlgorithm.build_model` and treat its KL term as zero. The exact lines are in the docstring of `src/dvae/model/gru_d.py`. `forward` already accepts `mode_selector`.

### SINDy

Brunton et al. 2016. Sparse polynomial dynamics, fit on the training trajectory and scored on test windows with a half-series free run. The default fitter is NumPy sequential thresholded least squares, so the container does not need a new package. `pip install 'research_dvae[sindy]'` (or `pysindy`) enables `--backend pysindy`.

```sh
python bin/eval_sindy_lorenz.py \
  --observation-process all_xyz \
  --missing-strategy observed_segments \
  --output-dir ./outputs/sindy_all_xyz
```

On Deigo:

```sh
bash scripts/slurm/evaluation/run_sindy_lorenz.sh
MASK_LABEL=Bernoulli_0.5 bash scripts/slurm/evaluation/run_sindy_lorenz.sh
```

`MASK_LABEL` has to be a mask that exists under the Lorenz data directory. The job writes one folder per condition with `evaluation_summary.yaml`, `equations.txt`, `limits.txt`, and plots styled by `aggregate_plot_style`.

Missing values: `observed_segments` (default) drops any row with a NaN and pools the contiguous runs, so the derivative is not taken across a gap. `interpolate` linearly fills each coordinate first; that is optimistic when gaps are long.

Partial observation is a separate limit. `only_x` / `only_xy` / `only_xz` are fit in the coordinates that remain. Lorenz63 needs the `xy` and `xz` terms, so a 1-D or 2-D polynomial ODE cannot represent the attractor. Expect a faithful sparse model on fully observed `all_xyz` and a clear drop when a coordinate is absent. Delay embedding is not applied.

`mse_auto`, `kld_auto`, and `spectrum_error_auto` are the same scores the OTF aggregate plots read (masked MSE, delay-embedding GMM KL with Lorenz tau=10, m=3, Hellinger spectrum error). Full-horizon MSE on Lorenz grows even for a correct field because of chaos. `mse_auto_short` (first 20 forecast steps), the spectrum error, and the KL are the comparisons that stay informative. SINDy is fit on the whole training trajectory, which is a stronger sample than the windowed RNN objective.

## XhroProper session paths

`XhroProper` reads `grok_output/<session>/intermediates/*.npz`. With neither `corpus` nor `data_root` set, the path stays the packet-loss tree:

`{data_dir}/xhro_packet_loss/grok_output/{dataset_label}_{mask_label}`

`mask_label` is `realtime` or `retrans` (`recovered` → `retrans`).

Multi (suntory) sessions have no variant suffix. In `[DataFrame]` set `corpus = multi` and `dataset_label` to the session folder (`xhro_01_XH015`). The default parent is `{data_dir}/suntory/xhro_dataset_v2/grok_output`. `mask_label` is ignored.

`data_root` replaces that parent: an absolute path is used as given, a relative path is joined onto `data_dir`. Pair it with `corpus = multi` when the Multi tree is not at the default relative path. Other datasets ignore both keys. `Xhro` and `XhroPacketLoss` keep their parquet loaders.


# Dynamical Variational Autoencoders A Comprehensive Review

This repository contains the code for:  
Dynamical Variational Autoencoders: A Comprehensive Review, **Foundations and Trends in Machine Learning**, 2021.  
[Laurent Girin](http://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/cv_en.html), [Simon Leglaive](https://sleglaive.github.io/index.html), [Xiaoyu Bie](https://xiaoyubie1994.github.io/), [Julien Diard](https://diard.wordpress.com/), [Thomas Hueber](http://www.gipsa-lab.grenoble-inp.fr/~thomas.hueber/), [Xavier Alameda-Pineda](http://xavirema.eu/)  
**[[arXiv](https://arxiv.org/abs/2008.12595)]** **[[Paper](https://www.nowpublishers.com/article/Details/MAL-089)]** **[[Project](https://team.inria.fr/robotlearn/dvae/)]** **[[Tutorial](https://dynamicalvae.github.io/)]**


<img src="./DVAE_overview.png" width="60%">


More precisely, this repo is a re-implementation of the following models in Pytorch:
- [VAE](https://arxiv.org/abs/1312.6114), Kingma et al., ICLR 2014
- [DKF](https://arxiv.org/abs/1609.09869), Krishnan et al., AAAI 2017
- [KVAE](https://arxiv.org/abs/1710.05741), Fraccaro et al., NeurIPS 2017
- [STORN](https://arxiv.org/abs/1411.7610), Bayer et al., arXiv 2014
- [VRNN](https://arxiv.org/abs/1506.02216), Chung et al., NeurIPS 2015
- [SRNN](https://arxiv.org/abs/1605.07571), Fraccaro et al., NeurIPS 2016
- [RVAE](https://arxiv.org/abs/1910.10942), Simon et al., ICASSP 2020
- [DSAE](https://arxiv.org/abs/1803.02991), Yingzhen et al. ICML 2018

> For the results we report at Interspeech 2021, please visit the **interspeech branch**  

> We don't report the results of KVAE since we haven't make it work in our experiments, we still provide the code for research purpose  

## Pretrained models

You could download all the pre-trained DVAE models [here](https://zenodo.org/record/6881571#.Ytp4gtLP05m) 


## Prerequest
- Pytorch >= 1.4.0
- [librosa](https://pypi.org/project/librosa/)
- [soundfile](https://pypi.org/project/SoundFile/)
- [pypesq](https://pypi.org/project/pypesq/)
- [pystoi](https://pypi.org/project/pystoi/)

> The PESQ value we report in our paper is a narrow-band PESQ value provide by [pypesq](https://github.com/vBaiCai/python-pesq) package. If you want to get a wide-band PESQ value, please use [pesq](https://github.com/ludlows/python-pesq) package instead

## Dataset
In this version, DVAE models support two differnt data structure:
- WSJ0, an audio speech data, we use the subset [ChiME2-WSJ0](https://catalog.ldc.upenn.edu/LDC2017S10) from [ChiME-Challenge](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/)
- Human3.6M, a 3D human motion data under license [here](http://vision.imar.ro/human3.6m/eula.php), the exponential map version can be download [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)

> If you want to use our models in other datasets, you can simply modify/re-write the dataloader and make minor changes in the training steps. Please remind that DVAE models accept data in the format of `(seq_len, batch_size, x_dim)`

To add new datasets:
1. 
1. 
1. In `dvae/learning_algo.py` within `def train` of `class LearningAlgorithm()`, inside the `for` loop,
```
 # Batch training
            for _, batch_data in enumerate(train_dataloader):
                batch_data = batch_data.to(self.device)
                if self.dataset_name == 'WSJ0':
                ...
```
add:
```
                elif self.dataset_name == 'your_dataset':
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = ...
                    recon_batch_data = ...
                    loss_recon = ...
```


## Train

We provide all configuration examples of the above models in `./confg`
```python
# Train on DVAE (for example)
python train_model.py --cfg ./config/speech/cfg_rvae_Causal.ini
python train_model.py --cfg ./config/motion/cfg_srnn.ini

# Train DVAE with schedule sampling, w/o. pretrained model
python train_model.py --ss --cfg ./confgi/speech/cfg_srnn_ss.ini --use_pretrain --pretrain_dict /PATH_PRETRAIN_DIR

# Train DVAE with schedule sampling, w. pretrained model
python train_model.py --ss --cfg ./confgi/speech/cfg_srnn_ss.ini

# Resume training
python train_model.py --cfg ./config/speech/cfg_rvae_Causal.ini --reload --model_dir /PATH_RELOAD_DIR
```

## Evaluation

```python
# Evaluation on speech data
python eval_wsj.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT --test_dir PATH_TO_TEST_DATA
python eval_wsj.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT --test_dir PATH_TO_TEST_DATA # schedule sampling

# Evaluation on human motion data
python eval_h36m.py --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT --test_dir PATH_TO_TEST_DATA
python eval_h36m.py --ss --cfg PATH_TO_CONFIG --saved_dict PATH_TO_PRETRAINED_DICT --test_dir PATH_TO_TEST_DATA # schedule sampling
```

## Bibtex
If you find this code useful, please star the project and consider citing:

```
@article{dvae2021,
  title={Dynamical Variational Autoencoders: A Comprehensive Review},
  author={Girin, Laurent and Leglaive, Simon and Bie, Xiaoyu and Diard, Julien and Hueber, Thomas and Alameda-Pineda, Xavier},
  journal={Foundations and Trends® in Machine Learning},
  year = {2021},
  volume = {15},
  doi = {10.1561/2200000089},
  issn = {1935-8237},
  number = {1-2},
  pages = {1-175}
}
```

```
@inproceedings{bie21_interspeech,
  author={Xiaoyu Bie and Laurent Girin and Simon Leglaive and Thomas Hueber and Xavier Alameda-Pineda},
  title={{A Benchmark of Dynamical Variational Autoencoders Applied to Speech Spectrogram Modeling}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={46--50},
  doi={10.21437/Interspeech.2021-256}
}
```


## Main results

For speech data, using:

- training dataset: wsj0_si_tr_s
- validation dataset: wsj0_si_dt_05
- test dataset: wsj0_si_et_05

| DVAE           |  SI-SDR(dB)  | PESQ | ESTOI |
| ----           |  ---------   | ---- | ---- |
| VAE            |      5.3     | 2.97 | 0.83 |
| DKF            |      9.3     | 3.53 | 0.91 |
| STORN          |      6.9     | 3.42 | 0.90 |
| VRNN           |      10.0    | 3.61 | 0.92 |
| SRNN           |      11.0    | 3.68 | 0.93 |
| RVAE-Causal    |      9.0     | 3.49 | 0.90 |
| RVAE-NonCausal |      8.9     | 3.58 | 0.91 |
| DSAE           |      9.2     | 3.55 | 0.91 |
| SRNN-TF-GM     |      -1.0    | 1.93 | 0.64 |
| SRNN-GM        |      7.8     | 3.37 | 0.88 |

For human motion data, using:

- training dataset: S1, S6, S7, S8, S9
- validation dataset: S5
- test dataset: S11

| DVAE           |  MPJPE (mm)  |
| ----           |  ---------   |
| VAE            |     48.69    |
| DKF            |     42.21    |
| STORN          |      9.47    |
| VRNN           |      9.22    |
| SRNN           |      7.86    |
| RVAE-Causal    |     31.09    |
| RVAE-NonCausal |     28.59    |
| DSAE           |     28.61    |
| SRNN-TF-GM     |     221.87   |
| SRNN-GM        |     43.98    |

More results can be found in **Chapter 13 Experiments** of our article.

## Contact

For any further questions, you can drop me an email via xiaoyu[dot]bie[at]inria[dot]fr

