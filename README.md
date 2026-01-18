# Reprot_DataMining_2025A_Mingzhuo

#Network-Based Topology Classification of IceCube Neutrino Events
## Dataset
In this project, I will use the publicly available IceCube Open Data provided in HDF5 format. The dataset contains pulse-level detector data recorded by Digital Optical Modules (DOMs) in the IceCube neutrino observatory. Each pulse includes the spatial position of the DOM, the arrival time of detected Cherenkov photons, and the collected charge. Monte Carlo truth labels are available for event topology, allowing supervised analysis.
The goal of this project is to classify IceCube neutrino events into different topological categories (track-like vs cascade-like) based on the detector response. Event topology reflects the underlying neutrino interaction channel and plays a crucial role in subsequent reconstruction tasks such as direction and energy estimation.
## Analysis Approach
Each neutrino event will be represented as a network, where nodes correspond to DOMs (or pulses) that record signals in the event and edges are defined based on spatial proximity and/or temporal adjacency between nodes.
This representation naturally connects the IceCube detector data to the network data analysis framework studied in the course. The constructed networks will be analyzed to capture their structural differences, which are expected to distinguish elongated track-like events from more localized cascade-like events.
Using the extracted network representations learnt from this course, I will formulate topology identification as a classification problem. A machine learning classifier will be trained to predict event topology from network-based features or network-aware representations. Model performance will be evaluated using standard classification metrics such as accuracy and ROC–AUC.

## Dependencies
