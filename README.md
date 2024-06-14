# Multimodal (2D/3D) change detection for natural disaster response </h1>

**This tutorial will highlight the capabilities of multimodal satellite imagery in addressing some of nowadays most impacting societal challenges using open source tools from CNES (French space agency).**

In anticipation of the future CO3D mission, the French space agency (CNES) is currently developing open source tools for large-scale 3D data processing. With the arrival of 3D data on a more frequent basis, new use-cases for such data sources are emerging and enabling its applicability to pressing challenges such as improved crisis management in the aftermath of natural disasters. Change detection methodologies and their application to natural disaster response constitute a long-standing field within the remote sensing community, with initiatives like The International Charter Space and Major Disasters, open data programs of commercial satellite providers or annotated datasets like xview2 enabling a growing number of researchers and practitioners to continuously improve existing solutions. However, until now, the availability of post-event imagery, and thus related algorithms, have mainly been limited to monoscopic acquisitions, whereas with this type of constellations, a growing number of 3D data should also cover areas impacted by climate-related hazards.

In this context, CNES is currently working on the hybridization of 2D and 3D data for multimodal change detection. The proposed tutorial will present and teach some of this work being carried out by CNES through a pipeline that can be used for rapid mapping and longer-term risk and recovery management in order to help improving and enriching common existing approaches.

During this tutorial, you will discover how to:
- generate Digital Surface Models from sets of stereo satellite images using the CARS 3D reconstruction library : https://github.com/CNES/cars
- extract the Digital Terrain Models from Digital Surface Models through the Bulldozer library and derive a final Digital Height Model based on the obtained results : https://github.com/cnes/bulldozer
- extract semantic information from the 2D imagery by using classification models and spectral indices
- combine 2D and 3D change indicators in order to quantify and localize the potentially changed areas on a map
- visualize, filter and extract information on these changes using the uncertainties provided by the tools