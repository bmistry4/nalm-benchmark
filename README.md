# Neural Arithmetic Logic Modules: Arithmetic Benchmarking Tasks
This repository is the official implementation for experiments in [A Primer for Neural Arithmetic Logic Modules](TODO),
providing the required files to generate Figures 14-19 corresponding to the Single Module Arithmetic Task.

This repo builds ontop of the codebase from **[Neural Arithmetic Units](https://openreview.net/forum?id=H1gNOeHKPS) 
by Andreas Madsen and Alexander Rosenberg Johansen**. 
The [original code](https://github.com/AndreasMadsen/stable-nalu) is by Andreas Madsen, who created the 
underlying framework used to create scripts to generate datasets, run experiments, and generate plots. 
**See their original README ([below](#neural-arithmetic-units))**.

## About 
The **Single Layer Task** is a benchmark task to evaluate the performance of Neural Arithmetic Logic Modules (NALMs) to 
perform a core arithmetic operation (add, subtract, times, or divide) on various synthetic numerical distributions. 
Specifically, modules use supervised learning to model an arithmetic relation: **y = a op b where op is either +, -, * or /**. 
Only the input values (a,b) and output (y) are provided, meaning the underlying operation is learnt. 


The implemented NALMs are listed below: 
- [NALU](https://arxiv.org/abs/1808.00508) 
- NAC+
- NAC*
- [iNALU](https://www.frontiersin.org/articles/10.3389/frai.2020.00071/full)
- [G-NALU](https://ieeexplore.ieee.org/document/8995315) (golden ratio NALU) 
- [NAU](https://ieeexplore.ieee.org/document/8995315)
- [NMU](https://ieeexplore.ieee.org/document/8995315)
- [NPU](https://arxiv.org/abs/2006.01681)
- [Real NPU](https://arxiv.org/abs/2006.01681)

## Create env
Generate a conda environment called nalu-env: 
`conda env create -f nalu-env.yml`

Install stable-nalu:
`python3 setup.py develop'`

## Recreating Experiments From the Paper:
First, create a csv file containing the threshold values for each range using 
<pre> Rscript <a href="export/single_layer_task/benchmark/generate_exp_setups.r">generate_exp_setups.r</a> </pre>

#### Generating plots consists of 3 stages
1. Run a shell script which calls the python script to _generate the tensorboard results_ over multiple seeds and ranges
    - `bash lfs_batch_jobs/single_layer_task/benchmarking/single_layer_benchmark.sh 0 24`
    - The *0 24* will run 25 seeds in parallel (i.e. seeds 0-24).
2. Call the python script to convert the tensorboard _results to a csv file_
    - `python3 export/simple_function_static.py --tensorboard-dir 
/data/nalms/tensorboard/<experiment_name>/ --csv-out /data/nalms/csvs/<experiment_name>.csv`
        - `--tensorboard-dir`: Directory containing the tensorboard folders with the model results
        - `--csv-out`: Filepath on where to save the csv result file
        - `<experiment_name>`: value of the experiment_name variable in the shell script used for step 1

3. Call the R script to convert the csv results to a _plot_ (saved as pdf)
    - <pre> Rscript <a href="export/single_layer_task/benchmark/plot_results.r">plot_results.r</a> None /data/nalms/plots/benchmark/sltr-in2/ benchmark_sltr op-add None benchmark_sltr_add </pre>
        - First arg: N/A
        - Second arg: Path to directory where you want to save the plot file
        - Third arg: Contributes to the plot filename. Use the Output value (see table below).
        - Forth arg: Arithmetic operation to create plot of (i.e. op-add, op-sub, op-mul, and op-div)
        - Fifth arg: N/A
        - Sixth arg: Lookup key (see table below) used to load relevant files and plot information
    

##### Experiment Meta-Information Table
| Figure | Experiment                      | Output value       |   Lookup key               |
|--------|---------------------------------|--------------------|----------------------------|
| 14     | Addition                        | benchmark_sltr     |   benchmark_sltr_add       |
| 15     | Subtraction                     | benchmark_sltr     |   benchmark_sltr_sub       |
| 16     | Multiplication                  | benchmark_sltr     |   benchmark_sltr_mul       |
| 17     | Division                        | benchmark_sltr     |   benchmark_sltr_div       |
| 18     | iNALU (input size 10)           | N/A (see below)    |   N/A (see below)          |
| 19     | NAU (input size 100)            | N/A (see below)    |   N/A (see below)          |


#### iNALU (input size 10) - Figure 18
Generate the tensorboard results and the csv file using the first two stages. 
To generate the plot, run:  
<pre> Rscript <a href="export/single_layer_task/benchmark/plot_results.r">plot_results.r</a> /data/nalms/csvs/benchmark/sltr-in10/add/ /data/nalms/plots/benchmark/sltr-in10/add/ iNALU op-add None</pre>

#### NAU (input size 100) - Figure 19
Generate the tensorboard results and the csv file using the first two stages. 
To generate the plot, run:  
<pre> Rscript <a href="export/single_layer_task/benchmark/plot_results_multiop.r">plot_results_multiop.r</a> /data/nalms/csvs/benchmark/sltr-in100 /data/nalms/plots/benchmark/sltr-in100/ NAU-in100-failures op-sub-add None nau-failure</pre>


---
# Neural Arithmetic Units

This code encompass two publiations. The ICLR paper is still in review, please respect the double-blind review process.

## Publications

#### SEDL Workshop at NeurIPS 2019

Reproduction study of the Neural Arithmetic Logic Unit (NALU). We propose an improved evaluation criterion of arithmetic tasks including a "converged at" and a "sparsity error" metric. Results will be presented at [SEDL|NeurIPS 2019](https://sites.google.com/view/sedl-neurips-2019/#h.p_vZ65rPBhIlB4). – [Read paper](http://arxiv.org/abs/1910.01888).

```bib
@inproceedings{maep-madsen-johansen-2019,
    author={Andreas Madsen and Alexander Rosenberg Johansen},
    title={Measuring Arithmetic Extrapolation Performance},
    booktitle={Science meets Engineering of Deep Learning at 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)},
    address={Vancouver, Canada},
    journal={CoRR},
    volume={abs/1910.01888},
    month={October},
    year={2019},
    url={http://arxiv.org/abs/1910.01888},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    eprint={1910.01888},
    timestamp={Fri, 4 Oct 2019 12:00:36 UTC}
}
```

#### ICLR 2020 (Spotlight)

Our main contribution, which includes a theoretical analysis of the optimization challenges with the NALU. Based on these difficulties we propose several improvements. **This is under double-blind peer-review, please respect our anonymity and reference https://openreview.net/forum?id=H1gNOeHKPS and not this repository!** – [Read paper](https://openreview.net/forum?id=H1gNOeHKPS).

```bib
@inproceedings{mnu-madsen-johansen-2020,
    author = {Andreas Madsen and Alexander Rosenberg Johansen},
    title = {{Neural Arithmetic Units}},
    booktitle = {8th International Conference on Learning Representations, ICLR 2020},
    volume = {abs/2001.05016},
    year = {2020},
    url = {http://arxiv.org/abs/2001.05016},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    arxivId = {2001.05016},
    eprint={2001.05016}
}
```

## Install

```bash
python3 setup.py develop
```

This will install this code under the name `stable-nalu`, and the following dependencies if missing: `numpy, tqdm, torch, scipy, pandas, tensorflow, torchvision, tensorboard, tensorboardX`.

## Experiments used in the paper

All experiments results shown in the paper can be exactly reproduced using fixed seeds. The `lfs_batch_jobs`
directory contains bash scripts for submitting jobs to an LFS queue. The `bsub` and its arguments, can be
replaced with `python3` or an equivalent command for another queue system.

The `export` directory contains python scripts for converting the tensorboard results into CSV files and
contains R scripts for presenting those results, as presented in the paper.

## Naming changes

As said earlier the naming convensions in the code are different from the paper. The following translations
can be used:

* Linear: `--layer-type linear`
* ReLU: `--layer-type ReLU`
* ReLU6: `--layer-type ReLU6`
* NAC-add: `--layer-type NAC`
* NAC-mul: `--layer-type NAC --nac-mul normal`
* NAC-sigma: `--layer-type PosNAC --nac-mul normal`
* NAC-nmu: `--layer-type ReRegualizedLinearPosNAC --nac-mul normal --first-layer ReRegualizedLinearNAC`
* NALU: `--layer-type NALU`
* NAU: `--layer-type ReRegualizedLinearNAC`
* NMU: `--layer-type ReRegualizedLinearNAC --nac-mul mnac`

## Extra experiments

Here are 4 experiments in total, they correspond to the experiments in the NALU paper.

```
python3 experiments/simple_function_static.py --help # 4.1 (static)
python3 experiments/sequential_mnist.py --help # 4.2
```

Example with using NMU on the multiplication problem:

```bash
python3 experiments/simple_function_static.py \
    --operation mul --layer-type ReRegualizedLinearNAC --nac-mul mnac \
    --seed 0 --max-iterations 5000000 --verbose \
    --name-prefix test --remove-existing-data
```

The `--verbose` logs network internal measures to the tensorboard. You can access the tensorboard with:

```
tensorboard --logdir tensorboard
```
