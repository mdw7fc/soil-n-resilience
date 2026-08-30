*********************************************************************************************************
Global data set of Monthly Irrigated and Rainfed Crop Areas around the year 2000 (MIRCA2000), version 1.1
*********************************************************************************************************

------ CONTENTS

The files in this folder represent the Maximum Monthly Cropped Area (MMCA),
i.e. the maximum monthly growing area of either irrigated, rainfed, 
or the sum of irrigated and rainfed crops altogether (crops 1-26)
in 5 arc-minute and 30 arc-minute grid cell resolution (data unit: hectare). 

The MIRCA2000-dataset is documented at: 
http://www2.uni-frankfurt.de/45218023/MIRCA.

This is version 1.1 of the data set. The difference to version 1.0 is the implementation of a bugfix to
ensure consistency of two major input data sets indicating cropland extent (Ramankutty et al., 2008) 
and harvested crop areas (Monfreda et al., 2008).


------ DATA FORMAT

The files (separately for irrigated (IRC), for rainfed crops (RFC), 
and for the sum of irrigated and rainfed crops (TOTAL)) 
are compressed (gnu-zip format) plain text files
in ESRI-ascii grid format that can be imported by most of the GIS-software
and other software that can handle gridded data, e.g. 
ESRI ArcViewGIS 3.x + Spatial Analyst extension, ESRI ArcGIS, ERDAS Imagine.
Each file contains a header of 6 rows. 
The following 4320 x 2160 grid cells contain the annual harvested area in hectare.
The cell order is from North to South and West to East.

The 4 files with areas of either irrigated or rainfed crops have the file names:
(1) 5 arc-minute grid:
	MAX_CROPPED_AREA_IRC_HA.ASC(.gz)
	MAX_CROPPED_AREA_RFC_HA.ASC(.gz)
(2) 30 arc-minute grid:
	max_cropped_area_irc_ha_30mn.asc(.gz)
	max_cropped_area_rfc_ha_30mn.asc(.gz).
	
The growing area is given in hectare (ha).


Extension as of September 2013:
The 2 files with areas of the total (sum) of irrigated and rainfed crops have the file names:
(file name including "annual" in order to represent the fact that, derived from monthly data,
only one value for a whole year is used)
(1) 5 arc-minute grid:
	MAX_CROPPED_AREA_TOTAL_ANNUAL_HA.ASC(.gz)
(2) 30 arc-minute grid:
	max_cropped_area_total_annual_ha_30mn.asc(.gz).


------ AUTHORS of the data compilation

Felix Portmann
Institute of Physical Geography (IPG), Goethe University Frankfurt am Main, Germany
portmann@em.uni-frankfurt.de

Stefan Siebert
Institute of Crop Science and Resource Conservation (INRES), University of Bonn, Germany
s.siebert@uni-bonn.de

September 2009
Extension September 2013