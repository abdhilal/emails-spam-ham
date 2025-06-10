# app.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# تهيئة تطبيق Flask
app = Flask(__name__)

# --- 1. تحديد مسار النموذج المُدرب ---
# هذا المسار نسبي لمكان ملف app.py ضمن هيكل المشروع
model_save_directory = "./trained_spam_model"

# تهيئة المتغيرات العالمية للنموذج والـ tokenizer
tokenizer = None
model = None
device = None

# وظيفة لتحميل النموذج والـ tokenizer
# يتم استدعاء هذه الوظيفة مرة واحدة عند بدء تشغيل التطبيق
def load_model_and_tokenizer():
    global tokenizer, model, device
    try:
        print(f"جاري تحميل النموذج والمُرمز من المسار: {model_save_directory}\n")
        tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
        # تصحيح: اسم الفئة الصحيح هو AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_save_directory)
        # اختيار الجهاز: GPU إذا كان متاحًا (cuda)، وإلا CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # وضع النموذج في وضع التقييم (inference mode)
        print(f"تم تحميل النموذج والمُرمز بنجاح على الجهاز: {device}.\n")
    except Exception as e:
        print(f"خطأ أثناء تحميل النموذج أو المُرمز: {e}")
        print("يرجى التأكد من أن جميع الملفات الضرورية (مثل config.json, model.safetensors) موجودة في المجلد المحدد.")
        # من المهم أن يفشل التطبيق عند بدء التشغيل إذا لم يتم تحميل النموذج
        exit()

# تحميل النموذج عند بدء تشغيل تطبيق Flask
# يجب أن يكون هذا الجزء خارج أي دالة ليعمل عند بدء التطبيق
with app.app_context():
    load_model_and_tokenizer()

# --- 2. تعريف وظيفة للتنبؤ ---
def predict_spam_ham(text_to_predict):
    # تحقق احتياطي من أن النموذج تم تحميله، على الرغم من أنه يجب أن يكون كذلك
    if tokenizer is None or model is None:
        print("خطأ: النموذج أو المُرمز غير محمل.")
        return "Model not loaded", 0.0

    # ترميز النص المدخل
    inputs = tokenizer(text_to_predict, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    # نقل المدخلات إلى نفس الجهاز الذي يوجد عليه النموذج (CPU/GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(): # تعطيل حساب التدرجات لتسريع الاستدلال وتوفير الذاكرة
        outputs = model(**inputs)
        logits = outputs.logits

    # تحويل الـ logits إلى احتمالات باستخدام softmax
    probabilities = torch.softmax(logits, dim=-1)

    # الحصول على الفئة المتوقعة (0 أو 1)
    predicted_class_id = probabilities.argmax().item()

    # تحويل ID إلى نص (وفقًا لما عرفته في تدريبك: 0=ham, 1=spam)
    id_to_label = {0: 'ham', 1: 'spam'}
    predicted_label = id_to_label[predicted_class_id]

    # الحصول على درجة الثقة (أقصى احتمال)
    confidence_score = probabilities.max().item()

    return predicted_label, confidence_score

# --- 3. تعريف نقطة نهاية API ---
# هذه النقطة هي التي ستتلقى طلبات POST مع نص البريد الإلكتروني
@app.route('/predict', methods=['POST'])
def predict():
    # استخراج بيانات JSON من الطلب
    data = request.get_json(force=True)
    email_text = data.get('email')

    # التحقق مما إذا تم توفير نص البريد الإلكتروني
    if not email_text:
        return jsonify({"error": "No email text provided in JSON payload"}), 400

    # استدعاء وظيفة التنبؤ
    label, confidence = predict_spam_ham(email_text)

    # إرجاع النتيجة كـ JSON
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4) # تقريب الثقة لـ 4 أرقام عشرية
    })

# نقطة نهاية بسيطة للصفحة الرئيسية للتحقق من أن التطبيق يعمل
@app.route('/')
def home():
    return "خدمة الكشف عن السبام تعمل! أرسل طلب POST إلى /predict مع نص البريد الإلكتروني في هيئة JSON."

# هذا الجزء للتشغيل المحلي فقط. Hugging Face Spaces سيستخدم Gunicorn لتشغيل التطبيق.
if __name__ == '__main__':
    # سيتم تشغيل التطبيق على المنفذ المحدد بواسطة متغير البيئة PORT، وإلا على المنفذ 5000
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
