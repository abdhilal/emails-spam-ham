# Dockerfile
# نبدأ بصورة بايثون خفيفة مبنية على Debian
FROM python:3.9-slim-buster

# تعيين دليل العمل داخل حاوية Docker
WORKDIR /app

# نسخ ملف requirements.txt إلى دليل العمل وتثبيت التبعيات
# هذا يضمن أن التبعيات يتم تثبيتها قبل نسخ باقي الكود
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ جميع ملفات المشروع المتبقية (app.py، مجلد النموذج، إلخ) إلى دليل العمل
COPY . .

# تحديد المنفذ الذي سيستمع إليه تطبيق Flask
# Hugging Face Spaces يتوقع عادةً المنفذ 7860
ENV PORT 7860

# الأمر الذي سيتم تشغيله عند بدء تشغيل الحاوية
# 'gunicorn' هو خادم WSGI الذي يقوم بتشغيل تطبيق Flask
# 'app:app' يعني تشغيل الكائن 'app' من الملف 'app.py'
# '-b 0.0.0.0:7860' يحدد العنوان والمنفذ للاستماع عليهما
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:7860"]