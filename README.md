# strelgen

This repository is the official github repository for the paper *Guiding neuro-symbolic scenario generation with spatio-temporal logic*.

 <!-- [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://cove-video.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-COVE-b31b1b.svg)](https://arxiv.org/abs/2406.08850)  -->

> **Guiding neuro-symbolic scenario generation with spatio-temporal logic**  
> [Lorenzo Bonin](https://scholar.google.com/citations?user=BI5BGaMAAAAJ&hl=it&oi=ao),
> [Francesco Giacomarra](https://scholar.google.com/citations?user=13zTza8AAAAJ&hl=it&oi=ao),
> [Luca Bortolussi](https://scholar.google.com/citations?user=p5ynADcAAAAJ&hl=it&oi=ao),
> [Jyotirmoy V. Deshmukh](https://scholar.google.com/citations?user=CwFX74MAAAAJ&hl=it&oi=ao),
> [Francesca Cairoli](https://scholar.google.com/citations?user=3s1GGlIAAAAJ&hl=it&oi=ao),


<p>

 The rapid advancement of autonomous driving (AD) technologies has outpaced the development of robust safety evaluation methods. Conventional testing relies on exposing AD systems to vast numbers of real-world traffic scenes—a brute-force approach that is prohibitively expensive and statistically ineffective at capturing the rare, safety-critical edge cases essential for validating real-world robustness. To address this fundamental limitation, we introduce *STRELGen*, a scalable framework for the targeted generation of safety-critical driving scenarios. *STRELGen* synergistically combines a multi-agent trajectory-generation diffusion model (DM) with Spatio-Temporal Logic (STREL) specifications that encode com- plex safety and realism properties through a highly interpretable formalism. Crucially, monitoring satisfaction levels of these specifications is differentiable, enabling gradient-based search. At inference time, we optimize directly over the DM’s latent space to maximize STREL formula satisfaction. The result is efficient generation of highly plausible yet safety-critical multi-agent scenarios that lie within the learned data distribution. *STRELGen* thus provides a flexible, interpretable, and powerful tool for stress-testing autonomous driving systems, moving beyond the limitations of brute-force data collection.

</p>

## News
- [2024.7.1] Paper is accepted by [AAMAS 2026](https://cyprusconferences.org/aamas2026/)!

## Initial Setup

**Step 1**: Download the code by cloning the repository:
```
git clone https://github.com/lorenzobonin/strelgen.git && cd strelgen
```

**Step 2**: Install required packages:
```
pip install -r requirements.txt
```

**Step 3**: Implement the [Argoverse 2 API](https://github.com/argoverse/av2-api) and access the [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html). Please see the [Argoverse 2 User Guide](https://argoverse.github.io/user-guide/getting_started.html).



## Joint Trajectory Prediction with Optimal Gaussian Diffusion

### Training Command
```sh
python train_diffnet_tb.py --root <Path to dataset> --train_batch_size 16 --val_batch_size 4 --test_batch_size 4 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --devices "4,5,6" --qcnet_ckpt_path <Path to QCNet checkpoint> --num_workers 4 --num_denoiser_layers 3 --num_diffusion_steps 100 --T_max 30 --max_epochs 30 --lr 0.005 --beta_1 0.0001 --beta_T 0.05 --diff_type opd --sampling ddim --sampling_stride 10 --num_eval_samples 6 --choose_best_mode FDE --std_reg 0.3 --check_val_every_n_epoch 3 --path_pca_s_mean 'pca/imp_org/s_mean_10.npy' --path_pca_VT_k 'pca/imp_org/VT_k_10.npy' --path_pca_V_k 'pca/imp_org/V_k_10.npy' --path_pca_latent_mean 'pca/imp_org/latent_mean_10.npy' --path_pca_latent_std 'pca/imp_org/latent_std_10.npy'
```
Below are the significant arguments related to our work:

- `--devices`: Specifies the GPUs you want to use.
- `--qcnet_ckpt_path`: Provides the path to the QCNet checkpoints.
- `--num_denoiser_layers`: Defines the number of layers in the diffusion network.
- `--num_diffusion_steps`: Sets the number of diffusion steps.
- `--max_epochs`: Determines the total number of training epochs.
- `--lr`: Sets the learning rate.
- `--beta_1`: Specifies the  $\beta_1$, the diffusion schedule parameter.
- `--beta_T`: Specifies the $\beta_T$, the diffusion schedule parameter.
- `--sampling_stride`: Defines the sampling stride for DDIM.
- `--num_eval_samples`: Indicates the number of evaluation samples.


### Validation Command
```sh
python val_diffnet.py --root <Path to dataset> --ckpt_path <Path to diffusion network checkpoint> --devices '5,' --batch_size 8 --sampling ddim --sampling_stride 10 --num_eval_samples 128 --std_reg 0.3 --path_pca_V_k 'pca/imp_org/V_k_10.npy' --network_mode 'val'
```

## Controllable Generation with STRELGen
```sh
python guided_strel_multiple.py --root /leonardo_scratch/fast/IscrC_ADGA/argoverse_data/ --ckpt_path lightning_logs/version_3/checkpoints/epoch=62-step=393624.ckpt --batch_size 16 --sampling ddim --sampling_stride 10 --num_eval_samples 1 --std_reg 0.3 --path_pca_V_k 'pca/imp_org/V_k_10.npy' --property ped_unsafe --lambda_reg 0.001 --lr 0.05 --max_steps 200 --num_samples 2


## Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. BibTex will be updated soon. Thanks for your support!

## Acknowledgements

This repository builds upon the original codebase developed by Yixiao Wang.
The backbone of this implementation is adapted from the following repository:

- https://github.com/YixiaoWang7/OptTrajDiff

We thank the authors for making their work publicly available.

Modifications and extensions have been made to support the contributions of this work.
