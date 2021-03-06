# Deep hierarchical sensory processing accounts for effects of arousal state on perceptual decision-making
by Lynn K.A. Sörensen, Sander M. Bohté, Heleen A. Slagter*, & H. Steven Scholte*


![](https://github.com/lynnsoerensen/Arousal_DCNN/blob/master/Figures/Figure1.png)
The figure is copied from this [preprint](https://www.biorxiv.org/content/10.1101/2021.05.19.444798v2). The example picture in B is licensed under CC BY-SA 2.0 and was adapted from [Flickr](https://farm2.staticflickr.com/1196/1089845176_c9b801237d_z.jpg).

### Overview
This is the code to reproduce the results of this [paper](https://www.biorxiv.org/content/10.1101/2021.05.19.444798v1). The repository is organized in three parts:

* the `asn` package for DCNNs for obtaining the ASN transfer function and the global gain modulation.
* the code to reproduce the results in the paper (`ModelPerformance`, `ModelAnalysis`)
* the code to reproduce the paper figures (`Figures`)



### Dependencies

Implementation for Keras (2.2.4) with a tensorflow backend (1.10).
All result files can be downloaded [here](https://osf.io/hwfvj). Please make sure to add the files to folder `Results` to reproduce the Figures.
The weights of the base models trained on ImageNet can be accessed [here](https://uvaauas.figshare.com/projects/Leveraging_spiking_deep_neural_networks_to_understand_neural_mechanisms_underlying_selective_attention/94406). 

