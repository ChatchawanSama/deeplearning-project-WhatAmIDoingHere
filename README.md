# 2WhatAmIDoingHere
## สมาชิก
1. นายธงไทย รุจิเวชวงษ์ 6310403982
2. นายชัชวาลย์ สามา 6310406272

# ขั้นตอนการใช้งานโค้ดในโปรเจกต์

- ## 1. ขั้นตอนการ clone
ทำการ Clone งานมาจาก Github Repository : deeplearning-project-WhatAmIDoingHere <br />
git clone https://github.com/ChatchawanSama/deeplearning-project-WhatAmIDoingHere.git

- ## 2. โหลด Dataset
โดยทำการโหลด Dataset จากลิงก์ Google Drive นี้ <br />
https://drive.google.com/file/d/1tTEBP4imIMSLULt5SPtnwira0FNCBAWO <br />
หลังจากที่ทำการโหลด Dataset มาจะได้ 4 โฟลเดอร์ดังนี้ <br />
* 2.1 heart_sound 
* 2.2 jwyy9np4gv-3 
* 2.3 LungDataset 
* 2.4 Respiratory_Sound_Database <br />
ซึ่งทั้ง 4 โฟลเดอร์นี้เป็น Dataset เกี่ยวกับเสียงหายใจบริเวณปอดที่ประกอบด้วยข้อมูลเสียงหายใจที่ Healthy และ Unhealthy มาจากแหล่งต่างๆ ดังนี้ <br />
* 2.1 https://www.kaggle.com/datasets/swapnilpanda/heart-sound-database
* 2.2 https://data.mendeley.com/datasets/jwyy9np4gv/3
* 2.3 https://www.youtube.com/playlist?list=PLT8Nd8-_R2iD_-GLfSQGJfsY8ZIs_p3kS
* 2.4 https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
จากนั้นให้ทำการจัดโฟลเดอร์ทั้ง 4 ที่ได้มาให้อยู่ในโฟลเดอร์เดียวกันกับที่เราทำการ clone ไว้ ดังรูป <br />
![image](https://user-images.githubusercontent.com/78588953/225033578-0755dc54-58d9-4053-a567-05f6c2c247cb.png)

- ## 3.เปิด Anaconda Prompt (Windows) ใน Mac เปิด Terminal ได้เลย
จากนั้นทำการสร้าง env และ activate env <br /><br />
```conda create -n ชื่อenv```<br />
```conda activate ชื่อenv```<br /><br />
จากนั้นทำการ install package ดังนี้<br /><br />
```conda install -c conda-forge jupyterlab```<br />
```conda install -c anaconda seaborn```<br />
```conda install -c conda-forge matplotlib```<br />
```conda install -c conda-forge librosa```<br />
```conda install -c anaconda scikit-learn```<br />
```conda install -c conda-forge tensorflow```<br />
```conda install -c anaconda pandas```<br />
```conda install -c anaconda numpy ```<br /><br />

- ## 4. เปิด Jupyter Notebook จากนั้นจะได้โค้ดและ Dataset ที่พร้อมใช้งาน
```jupyter notebook```<br /><br />
ทำการ run all ในไฟล์ 2WhatAmIdoingHere_ProjectV3.ipynb
![image](https://user-images.githubusercontent.com/78588953/225034082-73f1ce80-94e7-43cc-88d3-48b3b248ab2c.png)
