# AI CAD Converter v6

โปรแกรมต้นแบบสำหรับแปลงไฟล์รูปภาพหรือ PDF เป็นแบบ CAD ที่แก้ไขได้ โดยสร้าง
วัตถุ LWPOLYLINE, CIRCLE, HATCH, TEXT และ MTEXT จริงในไฟล์ DXF แทนการวางรูปภาพทับในแบบ

ผลลัพธ์หลักคือ DXF R2018 ซึ่งเปิดและแก้ไขได้ใน AutoCAD และ GstarCAD หากต้องการ
DWG โปรแกรมจะเรียก ODA File Converter ที่ติดตั้งอยู่บนเครื่องให้แปลง DXF เป็น DWG
โดยอัตโนมัติ

## ขอบเขต MVP ข้อ 1–6

| ข้อ | สถานะ | การทำงาน |
|---|---|---|
| 1. รับ PDF/รูปภาพ | พร้อมใช้ | รองรับ PDF หลายหน้าและ PNG/JPG/TIFF/WEBP/BMP/GIF |
| 2. ส่งออก CAD | พร้อมใช้ | สร้าง DXF ที่แก้ไขได้โดยตรง และสร้าง DWG ผ่าน ODA File Converter เมื่อเครื่องมีโปรแกรมดังกล่าว |
| 3. AI/ML/Neural learning | พร้อมใช้แบบเป็นขั้น | Auto decision เลือกวิธีประมวลผล, Random Forest เริ่มที่ feedback 5 งาน และ neural MLP เริ่มที่ 30 งาน |
| 4. Check and loop | พร้อมใช้ | สร้าง candidate หลายแบบ, render CAD กลับเป็นภาพ, เทียบ QA และวนจนถึงเป้าหมาย/คะแนนไม่ดีขึ้น/ครบจำนวนรอบ |
| 5. AI เป็นผู้กำหนด | พร้อมใช้ใน Auto mode | วิเคราะห์ blur, contrast, แสง, noise, สี และความหนาแน่นของแบบ แล้วเลือก preprocessing และจำนวนรอบเอง |
| 6. UX/UI Preview | พร้อมใช้ | Gradio UI แสดงต้นฉบับ, CAD reconstruction และ QA overlay ก่อนปล่อยไฟล์ดาวน์โหลด |

ตัว neural MLP ในเวอร์ชันนี้เป็นโมเดลจัดอันดับผลการแปลงจาก feedback ไม่ใช่โมเดล
มองภาพสำหรับจำแนกอุปกรณ์ การทำ Deep Vision เช่น YOLO/segmentation ต้องมีแบบที่ติด label
จริงก่อน จึงจะพัฒนาอย่างถูกต้องและพร้อมนำไปต่อในข้อ 7 (นับอุปกรณ์และ BOQ)

## ความสามารถที่มี

- รับไฟล์ PNG, JPG, TIFF, WEBP และ PDF ทุกหน้า
- ตรวจภาพ PNG/JPG ความละเอียดต่ำและขยาย 2x อัตโนมัติเพื่อช่วย OCR/เส้น โดยชดเชย
  สเกลตอนส่งออก CAD; หน้า UI เลือกโหมดเร็ว 1x, สมดุล 2x หรือรายละเอียดสูง 3x ได้
- อ่านเส้นและข้อความจาก PDF ที่เป็น vector โดยตรงก่อน เพื่อรักษาความคมและข้อความ
  selectable; ถ้าเป็น PDF scan จะกลับไปใช้ OpenCV + Tesseract OCR อัตโนมัติ
- อ่าน measurement scale ที่ฝังใน PDF วิศวกรรมโดยอัตโนมัติ และเลือก viewport หลัก
  เพื่อให้ขนาด CAD ตรงกับไฟล์ต้นฉบับมากขึ้น (สามารถปิดแล้วกำหนดสเกลเองได้)
- รวม line/curve segment ที่ต่อเนื่องกันเป็น LWPOLYLINE เพื่อลดเส้นแตกย่อย และแปลง
  solid fill เป็น HATCH บนเลเยอร์แยก
- เก็บข้อความ selectable จาก PDF เป็น MTEXT ที่แก้ไขได้ พร้อมตำแหน่ง สี และมุมหมุน
- ใช้ OpenCV สร้างภาพหลายแบบ เช่น Otsu, adaptive threshold, CLAHE, ตรวจเส้นสี,
  edge preserving, gamma correction, background normalization, black-hat, morphology และ sharpen
- Auto mode ตรวจคุณภาพไฟล์แล้วเลือกเทคนิคที่เหมาะสม โดยเก็บเหตุผลและคะแนนทุก
  รอบไว้ในไฟล์ report JSON
- ประมวลผล PDF ทีละหน้าและจำกัดภาพที่ถอดแล้วไว้ที่ 25 megapixels ต่อหน้า เพื่อลด
  ปัญหา memory เต็มบนบริการฟรี; CLI ปรับได้ด้วย `--max-page-megapixels`
- ทดลอง Tesseract OCR หลายวิธีปรับภาพ แล้วเลือกข้อความชุดที่มี confidence และ coverage
  ดีกว่าโดยอัตโนมัติ
- วนประมวลผลหลายรอบ แล้วให้คะแนนการสร้างวัตถุ CAD เทียบกับต้นฉบับ ระบบไม่หยุด
  ก่อนเวลาหากตรวจพบว่าต้นฉบับเบลอหรือ contrast ต่ำ
- ตรวจจับเส้นด้วย Hough transform และสัญลักษณ์/กรอบด้วย contour approximation;
  วงกลมจาก Hough ต้องมี contour กลมจริงรองรับ เพื่อป้องกันตารางและข้อความกลายเป็น
  วงกลมขนาดใหญ่ผิดปกติ
- ใช้ Tesseract OCR เพื่อแปลงข้อความเป็น TEXT ที่แก้ไขได้ โดยค่าเริ่มต้นคือภาษาไทย
  และอังกฤษ (tha+eng)
- สร้างภาพ QA สำหรับตรวจสอบผล: เขียว = ตรงกัน, แดง = มีในต้นฉบับแต่ยังไม่ถูกสร้าง,
  น้ำเงิน = มีใน CAD แต่ไม่อยู่ในต้นฉบับ
- มีหน้า UX/UI สำหรับอัปโหลดและปรับค่าการแปลง โดยซ่อนลิงก์ดาวน์โหลดไว้จนผู้ใช้
  ตรวจ Preview แล้วกดปุ่มยืนยัน
- รับคะแนนและคำแนะนำจากผู้ใช้ เก็บไว้ภายในเครื่อง เริ่มฝึก Random Forest หลังมี
  feedback 5 งาน และเปลี่ยนเป็น neural MLP แบบ 3 hidden layers หลังมี 30 งาน
- อ่าน measurement scale จาก PDF อัตโนมัติ หรือกำหนด Pixels per CAD unit เองเมื่อ
  ไฟล์ไม่มีข้อมูลสเกลฝังอยู่

## สิ่งที่ต้องติดตั้ง

ต้องมี Python 3.10 หรือใหม่กว่า และ Tesseract OCR

### Windows

สำหรับชุด PC แบบแยก ให้แตกไฟล์ ZIP แล้วดับเบิลคลิก `SETUP_PC.bat` หนึ่งครั้ง
จากนั้นใช้ `START_PC.bat` เพื่อเปิดโปรแกรมในเบราว์เซอร์ โดยมีคำแนะนำฉบับย่อใน
`README-PC-TH.md` ชุดติดตั้งจะสร้าง Python environment เฉพาะโปรแกรมและตรวจหา
Tesseract/ภาษาไทยให้อัตโนมัติ

หากใช้ source code โดยตรง สามารถติดตั้งเองได้ดังนี้:

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

สำหรับ Colab ให้เปิดไฟล์ notebooks/AI_CAD_Converter_Colab.ipynb แล้วอัปโหลด
ไฟล์ ZIP ของโปรแกรมตามเซลล์แรกได้ทันที ไม่จำเป็นต้องมี GitHub

## วิธีใช้หน้าโปรแกรม

1. รัน python app.py
2. เปิด URL ที่แสดงใน Terminal
3. เลือกรูปหรือ PDF
4. เปิด AI Auto mode ไว้เพื่อให้ระบบเลือกวิธีและจำนวนรอบเอง
5. สำหรับ PDF จาก CAD ให้เปิด **ใช้สเกลที่ฝังใน PDF อัตโนมัติ** หากไม่มีสเกลจึง
   ค่อยกำหนด Pixels per CAD unit ในตั้งค่าขั้นสูง
6. สำหรับ PNG/JPG เลือก **สมดุล 2x** ก่อน; ใช้ **ละเอียดสูง 3x** เฉพาะภาพที่มี
   ตัวอักษรหรือเส้นเล็กมาก เพราะจะใช้เวลามากขึ้น
7. เลือกภาษา OCR เป็น tha+eng และฟอนต์ที่รองรับไทย เช่น Arial.ttf
8. กด แปลงเป็น CAD และดู Preview
9. ตรวจภาพ Preview ซึ่งแสดงต้นฉบับ, CAD ที่สร้าง และ QA overlay
10. กด ยืนยัน Preview แล้วแสดงไฟล์ดาวน์โหลด
11. ดาวน์โหลด ZIP ซึ่งมี DXF, ภาพ QA, report JSON และ DWG ถ้าระบบสร้างสำเร็จ
12. เปิด DXF ใน AutoCAD/GstarCAD แล้วตรวจมิติ, เส้น, วงกลม และข้อความก่อนนำไปใช้ออกแบบ
13. ให้คะแนนผลลัพธ์และเขียนสิ่งที่ควรแก้ เพื่อสอนตัวจัดอันดับในครั้งต่อไป

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

หากต้องการบังคับใช้ค่า `--pixels-per-unit` เองและไม่อ่านสเกลจาก PDF:

~~~bash
python cli.py convert drawing.pdf --output outputs --pixels-per-unit 100 --no-auto-pdf-scale
~~~

บันทึก feedback จากไฟล์ report:

~~~bash
python cli.py feedback outputs/drawing_page_001_report.json --score 88 --accept --note "เส้นสีแดงบางเส้นยังขาด"
~~~

## ใช้งานบน GitHub

### ดาวน์โหลดชุด Windows PC

เปิดแท็บ Actions เลือก `Build Windows PC package` แล้วกด `Run workflow` เมื่อทำงาน
เสร็จให้ดาวน์โหลด artifact ชื่อ `AI-DWG-Converter-PC-Windows` แตก ZIP แล้วทำตาม
`README-PC-TH.md` ภายในไฟล์

### Codespaces: หน้า UI และ Preview แบบ interactive

หลังอัปโค้ดขึ้น repository ให้กด Code > Codespaces > Create codespace
ระบบจะติดตั้ง Python dependencies, OpenCV และ Tesseract ภาษาไทย/อังกฤษให้เองครั้งแรก

จาก Terminal ใน Codespaces ให้รัน:

~~~bash
./scripts/start_codespaces.sh
~~~

เปิดแท็บ Ports แล้วเลือก port 7860 เพื่อดูหน้าแปลงและ Preview ควรคง port เป็น Private
หากแบบมีข้อมูลโครงการที่ไม่ควรเปิดเผย

### GitHub Actions: แปลงแบบ batch

1. ใช้ private repository หากแบบเป็นข้อมูลภายใน
2. วาง PDF หรือรูปในโฟลเดอร์ input/ แล้ว commit ขึ้น repository
3. เปิดแท็บ Actions เลือก Convert CAD drawing แล้วกด Run workflow
4. ระบุ input_path เช่น input/drawing.pdf
5. เมื่อจบงาน ดาวน์โหลด artifact ชื่อ converted-cad-result

Action จะให้ DXF, QA preview และ report แต่ไม่มีหน้า Preview แบบ interactive และไม่
สร้าง DWG บน GitHub runner

## รันออนไลน์ด้วย Render

โครงการมี `Dockerfile` และ `render.yaml` ซึ่งติดตั้ง OpenCV, Tesseract ภาษาไทย/อังกฤษ
และเปิด Gradio ที่พอร์ตของ Render อัตโนมัติ ใช้ Dashboard ของ Render เลือก New >
Blueprint แล้วเชื่อม repository นี้ ระบบจะอ่าน `render.yaml` และสร้าง web service ให้

Free instance เหมาะสำหรับทดลอง ไม่ใช่ production และ filesystem เป็นแบบชั่วคราว
ดังนั้น feedback/model ที่อยู่ใน `/tmp/ai-cad-data` อาจหายเมื่อ service restart หากต้องการ
ให้ระบบเรียนรู้ต่อเนื่องในระยะยาว ต้องย้าย feedback ไป persistent database/object storage
หรือใช้ persistent disk ของแผนที่รองรับ

## หลักการเลือกผลที่ดีที่สุด

แต่ละภาพจะถูกปรับด้วยเทคนิคหลายแบบ แล้วแปลงเป็นวัตถุ CAD และ render กลับเป็นภาพ
เพื่อเทียบกับ ink mask ของต้นฉบับ คะแนนจะรวมความครอบคลุมของเส้น, ความแม่นยำของ
เส้น, IoU และความมั่นใจของ OCR ระบบจะหยุดก่อนกำหนดเมื่อได้คะแนนถึงเป้าหมาย หรือ
หยุดเมื่อครบจำนวนรอบที่ตั้งไว้

คะแนนนี้เป็นตัวช่วยเลือกผลลัพธ์ที่ใกล้รูปต้นฉบับที่สุด ไม่ใช่การรับรองว่ามิติจริง,
สเกล, สัญลักษณ์วิศวกรรม, line type หรือข้อความ OCR ถูกต้อง 100% งานวิศวกรรมต้อง
ตรวจและแก้ไขใน CAD ก่อนใช้งานจริงเสมอ

หากไฟล์ต้นฉบับไม่มีรายละเอียดเพราะเบลอ แตก หรือความละเอียดต่ำ ระบบสามารถเพิ่ม
contrast, ลด noise และเลือกผลที่ดีที่สุดจากข้อมูลที่ยังเหลืออยู่ แต่ไม่สามารถสร้างข้อมูล
วิศวกรรมที่ไม่มีอยู่ในต้นฉบับขึ้นมาอย่างน่าเชื่อถือได้

## โครงสร้างโครงการ

~~~text
cad_converter/
  decision.py       Input quality profiling and automatic strategy decisions
  preprocessing.py  OpenCV preprocessing variants
  vectorizer.py     LINE/CIRCLE/LWPOLYLINE detection and QA render
  pdf_vector.py     Native PDF paths, embedded scale, HATCH and editable MTEXT
  ocr.py            Tesseract text extraction
  exporter.py       DXF export and optional ODA DWG export
  learning.py       Random Forest / neural MLP feedback learning
  orchestrator.py   Iterative conversion pipeline
app.py              Gradio UI
cli.py              Batch / command-line interface
~~~
