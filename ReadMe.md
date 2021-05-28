# Deep hierarchical sensory processing accounts for effects of arousal state on perceptual decision-making
by Lynn K.A. Sörensen, Heleen A. Slagter, Sander M. Bohté, & H. Steven Scholte




### Overview
This is the code to reproduce the results of this [paper](https://www.biorxiv.org/content/10.1101/2021.05.19.444798v1). The repository is organized in three parts:

* the `asn` package for DCNNs for obtaining the ASN transfer function and the global gain modulation.
* the code to reproduce the results in the paper (`ModelPerformance`, `ModelAnalysis`)
* the code to reproduce the paper figures (`Figures`)



### Dependencies

Implementation for Keras (2.2.4) with a tensorflow backend (1.10).
All result files can be downloaded [here](https://osf.io/hwfvj/?view_only=6bc8a1c219dd4b4ba5185c3f86d3ed90). Please make sure to add the files to folder `Results` to reproduce the Figures.
The weights of the base models trained on ImageNet can be accessed here[here](https://uvaauas.figshare.com/projects/Leveraging_spiking_deep_neural_networks_to_understand_neural_mechanisms_underlying_selective_attention/94406). 

