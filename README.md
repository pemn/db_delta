# db_delta
Compare the migration table of a value between two datasets.
## Purpose
Common use cases are comparing blocks models or drillhole datasets.
## screenshots
### Graphic User Interface  
![screenshot1](assets/screenshot1.png?raw=true)  
  
### Result  
![screenshot2](assets/screenshot2.png?raw=true)  
## Parameters  
 - input: structured data in one of the supported file formats: xlsx, csv, shp (ESRI Shape)
 - key: (optional) field(s) unique to each record. If left blank, the row index will be used. Usually used when comparing block models by blockid.
 - group: (optional) the field which will be used to create the migration groups. Usually the lithology.
 - value: (optional) the sum field for the table. If blank the table will be a count of records. Usually `length` or `volume`.
 - condition: (optional) python syntax expression to restrict both datasets. Ex.: `length >= 4.0`. Usually left blank.
 - output_matrix: path to save the migration matrix in xlsx format.
 - output_records: path to save a raw table with all matched records. Usefull to debug. Usually left blank.
 - chart_bar: a 3d bar chart showing the scale of the changes by each group
 - chart_scatter: a 3d scatter chart showing where the changes occured in the space
## Repository
https://github.com/pemn/db_delta
