## Methods Supporting Interpretable and Explainable ML
This repository contains all experiments related to the Master's Thesis, with the ability to reproduce the obtained results.

The thesis was presented at [excel@fit 2025](https://excel.fit.vutbr.cz/sbornik/) student conference.

### Directory Structure 
```
Experiments/
├── ex1-model-vs-explanation-quality/
├── ex2-comparison-mnist-imagenet/
├── ex3-robustness-of-explanations/
├── ex4-ensembles-and-custom-extension/
├── side-experiments/                     # Additional experiments (outside thesis' scope)
└── requirements.txt
```

Each directory contains a specific experimental analysis, following the Master's Thesis outline. Each experiment is self-contained and can be executed independently. See individual notebooks or scripts for details and instructions.

### Results Reproduction
Jupyter notebooks in each specific experiment folder are organized in such a way that they contain all the necessary functions and utilities, which can be directly tweaked (in the notebook file). 

The simplest way to tweak the experiments is to create a Python virtual environment, install all the required dependencies, and that's all.

```
> python -m venv .venv
> source .venv/bin/activate        # On Unix/macOS
> .venv\\Scripts\\activate.bat     # On Windows

> pip install -r requirements.txt
```

To work with ImageNet experiments, download the [imagenet50-samples](https://drive.google.com/drive/folders/1IXOlElHJFRBFnM3mCLzUTPPRauiacT-h?usp=sharing) containing subset data.

---

### `Experiments/ex1-model-vs-explanation-quality/`
- **Model Quality vs. Explanation Quality**
- `ex1_model_vs_explanation_quality.ipynb`: Jupyter notebook analyzing how underfit, proper, and overfit models affect explanation quality.
- `ex1_train_kfold.py`: Script training multiple CNNs using K-fold cross-validation.
- `figures/`: Contains radar plots and saliency maps for each model.
- `models/`: Stores models being analyzed.

### `Experiments/ex2-comparison-mnist-imagenet/`
- **Dataset (MNIST vs. ImageNet) and XAI Method Comparison**
- `ex2a_mnist_xai.ipynb`: XAI method comparison on MNIST.
- `ex2b_imagenet_xai.ipynb`: XAI method comparison on ImageNet.
- `figures/`: Radar plots, method comparisons, sensitivity analysis.
- `imagenet50-samples/`: Preprocessed ImageNet subset.
- `models/`: Custom-trained CNN for MNIST.

### `Experiments/ex3-robustness-of-explanations/`
- **Robustness of Explanations under various conditions**
- `ex3a_robustness_under_noise.ipynb`: Stability analysis under Gaussian noise.
- `ex3b_input_invariance.ipynb`: Tests based on input-invariance transformations.
- `figures/`: Noise visualizations, invariance heatmaps, adversarial robustness.
- `models/`: Reused MNIST CNN.

### `Experiments/ex4-ensembles-and-custom-extension/`
- **Ensemble Explanation Methods and Custom Extension (AMS)**
- `ex4_ensembles.ipynb`: Experiments with explanation ensembling - analysis.
- `figures/`: Qualitative and quantitative comparisons of ensemble vs. singular methods.
- `imagenet50-samples/`: Preprocessed ImageNet subset.

---

### `Experiments/side-experiments/`
- `integrated-gradients/`: Notebook analyzing the Integrated Gradients attribution method.
- `manifolds/`: Code investigating the [manifold hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis) in the context of neural networks.
