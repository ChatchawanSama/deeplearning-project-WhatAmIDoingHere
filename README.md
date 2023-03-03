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
https://drive.google.com/drive/folders/1HA-Db1HNZUPeVe0KsJVexSyASwKIsh42?usp=sharing <br />
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
จากนั้นให้ทำการจัดโฟลเดอร์ทั้ง 4 ที่ได้มาให้อยู่ในโฟลเดอร์เดียวกันกับที่เราทำการ clone ไว้ <br />

- ## 3. เปิด Jupyter Notebook จากนั้นจะได้โค้ดและ Dataset ที่พร้อมใช้งาน

# ความคืบหน้า
ในตอนแรกคิดไว้ว่าจะใช้ข้อมูลเสียงที่แปลง Mel Spectrogram แล้วทำการใช้ SpecAugment คือการปิดปังบางส่วนของภาพ Mel Spectrogram ในการทำ Data Augmentation ผลที่ได้คือ model มีความ underfitting โดยที่ Accuracy Train : 60 % และ Accuracy Test : 60 % จึงทำการลองเปลี่ยนไปใช้ MFCCs แทน Mel Spectrogram และใช้วิธีการ SpecAugment ในการทำ Data Augmentation ผลที่ได้คือ model มีความแม่นยำมากขึ้นซึ่ง accuracy ตอน train และ test อยู่ที่ประมาณ 80 % และสามารถทำนายข้อมูลที่ไม่ได้มีความเกี่ยวข้องในช่วงพัฒนาได้แม่นยำ(ในบางครั้ง) อย่างเช่น เสียงจาก Youtube <br />
- ## ปัญหา
1.ถ้า Model ที่ใช้ MFCCs ไม่ได้ทำการใช้ SpecAugment เพื่อทำ Data Augmentation Accuracy ตอน Train และ Test จะอยู่ที่ 80 % แต่ไม่สามารถทำนายข้อมูลเสียงจาก Youtube ได้เลย อย่างเช่น ทำนายเป็น Healthy ทั้งหมด 18 ไฟล์ จาก 19 ไฟล์<br /><br />
2.ถ้าดูในไฟล์ .ipynb ของ V1 จะเห็นว่าข้อมูล MFCCs ที่ทำการใช้ SpecAugment ในการทำ Data Augmentation เมื่อนำไปทำนายเสียงจาก Youtube กลับทำนายผลออกมาได้ดีในระดับหนึ่ง แต่ไม่ใช่ว่าทุกครั้งที่ทำการ Train model จะทำนายผลข้อมูลจาก Youtube ได้ดีเหมือนดังในไฟล์ คิดว่าอาจเกิดจากการที่ข้อมูลมีการ shuffle แบบสุ่มทำให้ข้อมูลที่ใช้ในการ Train แต่ละครั้งแตกต่างกันทำให้ผลการทำนายที่ได้แตกต่างกัน ถึงจะมีการตั้ง seed ในการสุ่มไว้ แต่เนื่องจากเรามีการ run code ที่อยู่ใน cell ของการ shuffle ข้อมูลซ้ำๆ ดังนั้น การ shuffle ข้อมูลที่ทำให้ได้ผลดังในไฟล์ V1 ไม่ใช่การ shuffle ครั้งแรก และส่วนใหญ่ที่ทำการเทรนในแต่ละครั้งการทำนายในส่วนข้อมูลจาก Youtube ไม่ได้ดีมากนัก<br /><br />
3.การทำ SpecAugment เท่าที่ลองค้นหาพบว่ามีทำใช้กับ Mel Spectrogram เท่านั้น ยังไม่มีการทำกับ MFCCs ดังนั้นไม่สามารถการันตีว่า MFCCs สามารถใช้ SpecAugment ในการทำ Data Augment ได้ ตอนนี้จึงสามารถบอกได้แค่ว่าเป็นการเพิ่ม noise ในรูปภาพโดยการปิดปังบางส่วน<br /><br />
4.ในไฟล์ V2 (ปัจจุบัน) พบว่ามีข้อมูลเสียงบางส่วนสั้นเกินไป อาจเกิดจากการเขียนโค้ดในการดึงข้อมูลที่ผิด หรือ เป็นที่ตัวไฟล์เอง (ยังไม่ได้ตรวจสอบ)<br />
- ## ปัจจุบัน<br /><br />
1.กำลังทดลองวิธีในการทำ Data Augmentation อื่นๆ เช่น เพิ่ม noise ในข้อมูลเสียง , time shift กับทั้งไฟล์ .ipynb ที่ใช้ข้อมูลในการ train เป็น MFCCs และ ไฟล์ที่ใช้ Mel Spectrogram ในการ train ปัจจุบันยังพบว่ายังไม่สามารถทำนายเสียงที่มาจาก youtube ได้<br /><br />
2.สาเหตุที่ไม่สามารถทำนายข้อมูลจาก Youtube ได้ตอนนี้คิดว่า มาจาก 2 ประเด็นหลัง คือ 1.ข้อมูลในการเทรนไม่เพียงพอ ทำให้เมื่อไปทำนายกับเสียงที่มาจาก Youtube ที่มีคุณภาพแตกต่างจากข้อมูลที่ใช้เทรน ทำให้ ไม่สามารถ ทำนายได้ 2. มาจากปัญหาในข้อที่ 4<br /><br />
- SpecAugment(ref :https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52 )<br />
