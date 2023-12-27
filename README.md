## 📌 Description
### db_delta
Compare the migration table of a value between two datasets.
### db_buildup
Generate a simpler table of the difference and a waterfall+candlestick chart.

## 📝 Parameters
name|optional|description
---|---|------
input|❎|structured data in one of the supported file formats: xlsx, csv, shp (ESRI Shape)
key|☑️|(see notes) field(s) unique to each record. If left blank, the row index will be used. Usually used when comparing block models by blockid.
group|❎| the field which will be used to create the migration groups. Usually the lithology.
value|☑️|the sum field for the table. If blank the table will be a count of records. Usually `length` or `volume`.
condition|❎| python syntax expression to restrict both datasets. Ex.: `length >= 4.0`. Usually left blank.
output_matrix|☑️|path to save the migration matrix in xlsx format.
output_records|☑️| path to save a raw table with all matched records. Usefull to debug. Usually left blank.

## 📚 Examples
### Graphic User Interface  
![screenshot1](https://github.com/pemn/assets/blob/main/db_delta1.png?raw=true)
  
### Result  
![screenshot1](https://github.com/pemn/assets/blob/main/db_delta2.png?raw=true)

## 📓 Notes
If both datasets do not have the same number of records, the key parameter will be required.  
In case of block models, the script bm_flag_ijk can be used to create a suitable key.

## 🧩 Compatibility
distribution|status
---|---
![winpython_icon](https://github.com/pemn/assets/blob/main/winpython_icon.png?raw=true)|✔
![vulcan_icon](https://github.com/pemn/assets/blob/main/vulcan_icon.png?raw=true)|✔
![anaconda_icon](https://github.com/pemn/assets/blob/main/anaconda_icon.png?raw=true)|❌
## 🙋 Support
Any question or problem contact:
 - paulo.ernesto
## 💎 License
Apache 2.0
Copyright ![vale_logo_only](https://github.com/pemn/assets/blob/main/vale_logo_only_r.svg?raw=true) Vale 2023
