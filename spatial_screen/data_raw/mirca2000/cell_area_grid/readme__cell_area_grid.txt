*********************************************************************************************************
Global data set of Monthly Irrigated and Rainfed Crop Areas around the year 2000 (MIRCA2000), version 1.1
*********************************************************************************************************

------ CONTENTS

The files in this folder represent the grid cell area for each 5 arc-minute and 30 arc-minute grid cell 
(0.5 degree, aggregated from 5 arc-minute grid cells via ArcGIS) (data unit: hectare).

The MIRCA2000-dataset is documented at: 
http://www2.uni-frankfurt.de/45218023/MIRCA.

This is version 1.1 of the data set. The difference to version 1.0 is the implementation of a bugfix to
ensure consistency of two major input data sets indicating cropland extent (Ramankutty et al., 2008) 
and harvested crop areas (Monfreda et al., 2008).


------ DATA FORMAT

The files (one per resolution) are compressed (gnu-zip format) plain text files
in ESRI-ascii grid format that can be imported by most of the GIS-software
and other software that can handle gridded data, e.g. 
ESRI ArcViewGIS 3.x + Spatial Analyst extension, ESRI ArcGIS, ERDAS Imagine.
Each file contains a header of 6 rows. 
The following 4320 x 2160 grid cells contain the annual harvested area in hectare.
The cell order is from North to South and West to East.

The 2 files have the file names:
(1) 5 arc-minute grid:  cell_area_ha_05mn.asc(.gz)
(2) 30 arc-minute grid: cell_area_ha_30mn.asc(.gz).

The cell area is given in hectare (ha), and was computed using an equal-area projection.


------ AUTHORS of the data compilation

Felix Portmann
Institute of Physical Geography (IPG), Goethe University Frankfurt am Main, Germany
portmann@em.uni-frankfurt.de

Stefan Siebert
Institute of Crop Science and Resource Conservation (INRES), University of Bonn, Germany
s.siebert@uni-bonn.de

September 2009
Update September 2013: new URL for MIRCA2000 documentation