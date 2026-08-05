# AI CAD Converter

โปรแกรมต้นแบบสำหรับแปลงไฟล์รูปภาพหรือ PDF เป็นแบบ CAD ที่แก้ไขได้ โดยสร้าง
วัตถุ LINE, CIRCLE, LWPOLYLINE และ TEXT จริงในไฟล์ DXF แทนการวางรูปภาพทับในแบบ

ผลลัพธ์หลักคือ DXF R2018 ซึ่งเปิดและแก้ไขได้ใน AutoCAD และ GstarCAD หากต้องการ
DWG โปรแกรมจะเรียก ODA File Converter ที่ติดตั้งอยู่บนเครื่องให้แปลง DXF เป็น DWG
โดยอัตโนมัติ

## ความสามารถที่มี

- รับไฟล์ PNG, JPG, TIFF, WEBP และ PDF ทุกหน้า
- อ่านเส้นและข้อความจาก PDF ที่เป็น vector โดยตรงก่อน เพื่อรักษาความคมและข้อความ
  selectable; ถ้าเป็น PDF scan จะกลับไปใช้ OpenCV + Tesseract OCR อัตโนมัติ
- ใช้ OpenCV สร้างภาพหลายแบบ เช่น Otsu, adaptive threshold, CLAHE, ตรวจเส้นสี,
  edge preserving, gamma correction, morphology และ sharpen
- วนประมวลผลหลายรอบ แล้วให้คะแนนการสร้างวัตถุ CAD เทียบกับต้นฉบับ
- ตรวจจับเส้นด้วย Hough transform, วงกลมด้วย Hough circles และสัญลักษณ์/กรอบด้วย
  contour approximation
- ใช้ Tesseract OCR เพื่อแปลงข้อความเป็น TEXT ที่แก้ไขได้ โดยค่าเริ่มต้นคือภาษาไทย
  และอังกฤษ (tha+eng)
- สร้างภาพ QA สำหรับตรวจสอบผล: เขียว = ตรงกัน, แดง = มีในต้นฉบับแต่ยังไม่ถูกสร้าง,
  น้ำเงิน = มีใน CAD แต่ไม่อยู่ในต้นฉบับ
- มีหน้า UX/UI สำหรับอัปโหลดและปรับค่าการแปลง โดยซ่อนลิงก์ดาวน์โหลดไว้จนผู้ใช้
  ตรวจ Preview แล้วกดปุ่มยืนยัน
- รับคะแนนและคำแนะนำจากผู้ใช้ เก็บไว้ภายในเครื่อง และเริ่มฝึกโมเดลจัดอันดับ
  candidate หลังมี feedback อย่างน้อย 5 งาน
- กำหนดสเกลได้ ปัจจุบันค่าเริ่มต้นคือ 1 pixel = 1 CAD unit

## สิ่งที่ต้องติดตั้ง

ต้องมี Python 3.10 หรือใหม่กว่า และ Tesseract OCR

### Windows

~~~powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
~~~

ติดตั้ง Tesseract OCR และตรวจสอบว่าคำสั่ง tesseract อยู่ใน PATH จากนั้นตรวจภาษา:

~~~powershell
tesseract --list-langs
~~~

ควรเห็น eng และ tha หากไม่มี tha ให้ติดตั้งไฟล์ภาษาไทยของ Tesseract หรือเปลี่ยนช่อง
OCR languages ในหน้าโปรแกรมเป็น eng ชั่วคราว

### Ubuntu หรือ Google Colab

~~~bash
sudo apt-get update -qq
sudo apt-get install -y tesseract-ocr tesseract-ocr-tha
pip install -r requirements.txt
python app.py
~~~

สำหรับ Colab ให้เปิดไฟล์ notebooks/AI_CAD_Converter_Colab.ipynb แล้วแก้
REPOSITORY_URL เป็น URL ของ repository นี้

## วิธีใช้หน้าโปรแกรม

1. รัน python app.py
2. เปิด URL ที่แสดงใน Terminal
3. เลือกรูปหรือ PDF
4. ตั้ง Pixels per CAD unit เป็น 1.0 หากยังไม่มีขนาดอ้างอิง
5. เลือกภาษา OCR เป็น tha+eng
6. เลือกฟอนต์ CAD ที่มีอักษรไทย เช่น Arial.ttf ถ้าต้องการแสดงภาษาไทย
7. กด แปลงเป็น CAD และดู Preview
8. ตรวจภาพ Preview ซึ่งแสดงต้นฉบับ, CAD ที่สร้าง และ QA overlay
9. กด ยืนยัน Preview แล้วแสดงไฟล์ดาวน์โหลด
10. ดาวน์โหลด ZIP ซึ่งมี DXF, ภาพ QA, report JSON และ DWG ถ้าระบบสร้างสำเร็จ
11. เปิด DXF ใน AutoCAD/GstarCAD แล้วตรวจมิติ, เส้น, วงกลม และข้อความก่อนนำไปใช้ออกแบบ
12. ให้คะแนนผลลัพธ์และเขียนสิ่งที่ควรแก้ เพื่อสอนตัวจัดอันดับในครั้งต่อไป

## การสร้าง DWG

โปรแกรมสร้าง DXF โดยตรง เพราะ DXF เป็นรูปแบบเปิดสำหรับแลกเปลี่ยนข้อมูล CAD ส่วน
DWG เป็นรูปแบบเฉพาะ จึงต้องติดตั้ง ODA File Converter หรือใช้คำสั่ง Save As ใน
AutoCAD/GstarCAD

เมื่อติดตั้ง ODA File Converter แล้ว ให้เลือก Also create DWG ในหน้าโปรแกรม หรือใช้
คำสั่ง:

~~~bash
python cli.py convert drawing.pdf --output outputs --dwg --oda-path "C:/path/to/ODAFileConverter.exe"
~~~

หากไม่ติดตั้ง ODA โปรแกรมจะยังส่งออก DXF ที่แก้ไขได้เสมอ และรายงานจะแจ้งเหตุผลที่
ยังสร้าง DWG ไม่ได้

## ใช้งานจาก Command Line

~~~bash
python cli.py convert drawing.pdf --output outputs --ocr-languages tha+eng --passes 3 --zip
~~~

บันทึก feedback จากไฟล์ report:

~~~bash
python cli.py feedback outputs/drawing_page_001_report.json --score 88 --accept --note "เส้นสีแดงบางเส้นยังขาด"
~~~

## หลักการเลือกผลที่ดีที่สุด

แต่ละภาพจะถูกปรับด้วยเทคนิคหลายแบบ แล้วแปลงเป็นวัตถุ CAD และ render กลับเป็นภาพ
เพื่อเทียบกับ ink mask ของต้นฉบับ คะแนนจะรวมความครอบคลุมของเส้น, ความแม่นยำของ
เส้น, IoU และความมั่นใจของ OCR ระบบจะหยุดก่อนกำหนดเมื่อได้คะแนนถึงเป้าหมาย หรือ
หยุดเมื่อครบจำนวนรอบที่ตั้งไว้

คะแนนนี้เป็นตัวช่วยเลือกผลลัพธ์ที่ใกล้รูปต้นฉบับที่สุด ไม่ใช่การรับรองว่ามิติจริง,
scale, สัญลักษณ์วิศวกรรม, line type หรือข้อความ OCR ถูกต้อง 100% งานวิศวกรรมต้อง
ตรวจและแก้ไขใน CAD ก่อนใช้งานจริงเสมอ

## โครงสร้างโครงการ

~~~text
cad_converter/
  preprocessing.py  OpenCV preprocessing variants
  vectorizer.py     LINE/CIRCLE/LWPOLYLINE detection and QA render
  ocr.py            Tesseract text extraction
  exporter.py       DXF export and optional ODA DWG export
  learning.py       Local feedback learning
  orchestrator.py   Iterative conversion pipeline
app.py              Gradio UI
cli.py              Batch / command-line interface
~~~
