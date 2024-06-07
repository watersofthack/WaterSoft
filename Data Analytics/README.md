# Hydroinformatics Training - WaterSoftHack 2024


This repository serves as a landing page and introduction of HydroSuite: A collection of web-based software for the development and analysis of hydrologic workflows designed by the Hydroinformatics Lab at the University of Iowa. These libraries can be used for:

* Development of web applications.
* In-browser fast analysis.
* Enhancement of existing tools.

The libraries shown throughout this material aim to highlight the use of web-based systems and showcase the potential implementation and improvements to the current hydrological needs.

## Introduction

HydroSuite addresses the needs of hydrological sciences by providing state-of-the-art tools developed using web environments, web applications, and data sources, thereby creating a comprehensive application space. It consists of the following libraries:

|Library|Overview|Functionalities|
|-------|--------|-----|
|[HydroLang](https://github.com/uihilab/HydroLang)|Library for data acquisition, analysis, and visualization directly on the client side.|Data, Analysis, Visualization|
|[HydroCompute](https://github.com/uihilab/HydroCompute)|Library for high performance execution of functions and workflows on the client side.|Analysis, Computation|
|[HydroRTC](https://github.com/uihilab/HydroRTC)|Library for data transmission between servers and clients, allowing S2P and P2P communication.|Data, Analysis|

The libraries are publicly available with several examples highlighting their usage to develop end-to-end workflows. Moreover, they are released both as [npm](https://www.npmjs.com/) packages, or can be modified by  the end user by downloading the code directly from the public repository.


## Training

Each tutorial has been crafted using online free sandbox environments, enabling the deployment and modification of code without the necessity of hosting space. These tutorials run directly on the resources of the device running the browser and can be accessed from the following platforms:

|Tutorial|Link|
|--------|----|
|**HydroLang**|https://hydroinformatics.uiowa.edu/tutorials/hydrolang/|
|**HydroCompute**|https://hydroinformatics.uiowa.edu/tutorials/hydrocompute/|
|**HydroRTC**|https://hydroinformatics.uiowa.edu/tutorials/hydrortc/|


## Data Development
The examples that are shown within this tutorial come from the following sources:

- Data from governmental/non-governmental institutions throughout the world. Examples [here](https://waterdata.usgs.gov/blog/api_catalog/).
- Data from analysis/reanalysis outputs from statistically or physically derived models. Example [here](https://docs.unidata.ucar.edu/netcdf-c/current/examples1.html#example_simple_xy).
- Data the users might have on their own systems.

The data will be mostly accessed through API calls to different sources, and connection with saved data in the UIowa Hydroinformatics Lab's servers.

Each training module contains multiple examples showcasing the use of each library, with the final example demonstrating the utilization of all.

## Community

The highlighted libraries represent a subset of a larger collection of tools and acquisition systems designed to address various water-related challenges. Users and trainees are encouraged to utilize the tools they are already comfortable with for the development of case studies and projects.

These tools have been created to enhance and strengthen existing robust libraries, contributing to the development of a more comprehensive and resilient analysis.


## References
Erazo Ramirez, C., Sermet, Y., Molkenthin, F., & Demir, I. (2022). HydroLang: An open-source web-based programming framework for hydrological sciences. Environmental Modelling & Software, 157, 105525. https://doi.org/10.1016/j.envsoft.2022.105525

Erazo Ramirez, C., Sermet, Y., & Demir, I. (2023). HydroLang markup language: Community-driven web components for hydrological analyses. Journal of Hydroinformatics, 25(4), 1171-1187. https://doi.org/10.2166/hydro.2023.149

Erazo Ramirez, C., Sermet, Y., & Demir, I. (2024). HydroCompute: An open-source web-based computational library for hydrology and environmental sciences. Environmental Modelling & Software, 175, 106005. https://doi.org/10.1016/j.envsoft.2024.106005

Erazo Ramirez, C., Sermet, Y., Shahid, M., & Demir, I. (2024). HydroRTC: A web-based data transfer and communication library for collaborative data processing and sharing in the hydrological domain. Environmental Modelling & Software, 106068. https://doi.org/10.1016/j.envsoft.2024.106068

Ewing, G., Erazo Ramirez, C., Vaidya, A., & Demir, I. (2024). Client-side web-based model coupling using basic model interface for hydrology and water resources. Journal of Hydroinformatics, 26(2), 494-502. https://doi.org/10.2166/hydro.2024.212

