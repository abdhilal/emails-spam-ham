# run_model.py

# الخطوة 0: استيراد المكتبات الضرورية
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

print("تم استيراد جميع المكتبات بنجاح.\n")

# --- 1. تحديد مسار النموذج المُدرب ---
# تأكد أن هذا المسار صحيح ويشير إلى المجلد الذي حفظت فيه الملفات (config.json, model.safetensors, etc.)
model_save_directory = "C:\\Users\\MSI\\Desktop\\project AI\\emails\\trained_ham_model\\trained_spam_model"

# التحقق مما إذا كان المجلد موجودًا
if not os.path.isdir(model_save_directory):
    print(f"خطأ: المجلد '{model_save_directory}' غير موجود. يرجى التأكد من المسار الصحيح.")
    exit() # إيقاف البرنامج إذا لم يتم العثور على المجلد

print(f"جاري تحميل النموذج والمُرمز من المسار: {model_save_directory}\n")

# --- 2. تحميل المُرمز (Tokenizer) والنموذج (Model) ---
try:
    tokenizer = AutoTokenizer.from_pretrained(model_save_directory)
    model = AutoModelForSequenceClassification.from_pretrained(model_save_directory)
    print("تم تحميل النموذج والمُرمز بنجاح.\n")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل النموذج أو المُرمز: {e}")
    print("يرجى التأكد من أن جميع الملفات الضرورية (config.json, model.safetensors, tokenizer.json, vocab.txt, etc.) موجودة في المجلد المحدد.")
    exit()

# --- 3. تحديد الجهاز (CPU أو GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # وضع النموذج في وضع التقييم (inference mode)
print(f"سيتم استخدام الجهاز: {device} للاستدلال.\n")

# --- 4. تعريف وظيفة للتنبؤ ---
def predict_spam_ham(text_to_predict):
    # ترميز النص
    inputs = tokenizer(text_to_predict, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    # نقل المدخلات إلى نفس الجهاز الذي يوجد عليه النموذج
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
    
    # الحصول على درجة الثقة
    confidence_score = probabilities.max().item()

    return predicted_label, confidence_score

# --- 5. اختبار النموذج على رسائل جديدة ---
print("--- اختبار النموذج على رسائل بريد إلكتروني جديدة ---")

new_emails =  [
    # --- Ham Emails (50 messages) ---
    "Hi team, just a reminder about our stand-up meeting at 9 AM tomorrow. Please be prepared.", # ham
    "Please find attached the updated project timeline for your review. Let me know if you have any questions.", # ham
    "Hello John, I've completed the first draft of the report. Would you like to review it this afternoon?", # ham
    "Regarding your inquiry, we have processed your refund. It should appear in your account within 3-5 business days.", # ham
    "Confirming our appointment for Tuesday, June 10th, at 2:00 PM. Looking forward to it.", # ham
    "Could you please send me the latest sales figures for Q1? I need them for the presentation.", # ham
    "Just a quick note to say thank you for your help with the recent client issue. It was greatly appreciated.", # ham
    "Reminder: Your library book is due back on June 15th. Please return or renew it to avoid late fees.", # ham
    "Dear applicants, we've received your application and will be in touch shortly regarding the next steps.", # ham
    "I'm out of office until June 12th. For urgent matters, please contact Sarah at sarah.smith@example.com.", # ham
    "New team member alert! Please welcome Alex Miller, joining us as a Senior Software Engineer.", # ham
    "Your order #12345 has been shipped! You can track its progress at this link: [tracking link].", # ham
    "Attached is the agenda for our board meeting next week. Please review it before then.", # ham
    "Hi Mark, I'm having trouble with the VPN connection. Can you assist me?", # ham
    "Don't forget the company picnic this Saturday at City Park from 11 AM to 3 PM. Food and games provided!", # ham
    "Thank you for attending the webinar. Here is a link to the recording and presentation slides.", # ham
    "Please update your password for enhanced security. You can do so through your profile settings.", # ham
    "Just wanted to follow up on the proposal we discussed last week. Are you still interested?", # ham
    "Your monthly utility bill is now available. You can view and pay it online at [billing portal link].", # ham
    "Wishing you a very happy birthday! Enjoy your special day.", # ham
    "Can we reschedule our sync-up call to Thursday instead of Wednesday?", # ham
    "Review of Q2 performance will be held next Friday. Please bring your team's metrics.", # ham
    "The network maintenance is scheduled for Sunday night from 10 PM to 2 AM. Expect some downtime.", # ham
    "Good morning, I'll be late to work today due to a personal emergency. I'll inform you when I'm on my way.", # ham
    "Regarding the invoice, could you clarify the discrepancy in the amount?", # ham
    "New article published: 'The Future of AI in Healthcare.' Read it here: [blog link].", # ham
    "Your appointment with Dr. Evans is confirmed for tomorrow at 11 AM.", # ham
    "Please RSVP for the holiday party by December 10th.", # ham
    "The new printer drivers have been installed. You can now use the new network printer.", # ham
    "Seeking volunteers for our annual charity drive. Your support makes a difference!", # ham
    "Check out our latest product updates and new features. [Link to release notes].", # ham
    "Confirming receipt of your documents. We will review them and get back to you soon.", # ham
    "How was your vacation? Hope you had a great time!", # ham
    "Could you send me the contact information for the new vendor?", # ham
    "Your feedback is important! Please take a moment to complete our customer satisfaction survey.", # ham
    "Reminder: All expense reports for this month are due by the 25th.", # ham
    "The new security policy will be enforced starting next Monday. Please review it carefully.", # ham
    "I've shared the Google Drive folder with you. Let me know if you have access.", # ham
    "Happy Friday! Have a wonderful weekend.", # ham
    "Please bring your laptop fully charged to the training session tomorrow.", # ham
    "Your online banking statement is ready to view. Login to your account.", # ham
    "We regret to inform you that your application was not successful at this time.", # ham
    "Important safety notice: Emergency drills will be conducted next month.", # ham
    "Thank you for your patience during the system upgrade. Everything should be back to normal now.", # ham
    "The agenda for the upcoming community meeting is now available.", # ham
    "Just a quick note about the holiday schedule for the upcoming public holiday.", # ham
    "Can we set up a quick chat next week to discuss progress on the Smith project?", # ham
    "Your latest performance review summary is ready for your review.", # ham
    "The building will be closed for maintenance on July 4th.", # ham
    "Remember to lock your workstation when you step away.", # ham

    # --- Spam Emails (50 messages) ---
    "CONGRATULATIONS! YOU'VE WON A LUXURY CRUISE TRIP! CLICK HERE TO CLAIM YOUR PRIZE IMMEDIATELY!", # spam
    "URGENT ACTION REQUIRED: Your PayPal account has been limited due to suspicious activity. Verify your identity now!", # spam
    "LOSE 10 KILOS IN 7 DAYS! Revolutionary new diet pill with GUARANTEED RESULTS. Order now!", # spam
    "Your Amazon account is on hold! Click here to update your payment information or your order will be cancelled.", # spam
    "CLAIM YOUR UNCLAIMED FUNDS! Millions of dollars are waiting for you. Provide your bank details to release them.", # spam
    "Exclusive Offer: Get a FREE iPhone 15! Just pay for shipping and handling. Limited stock!", # spam
    "Nigerian Prince needs YOUR help to transfer millions! Get a percentage for your assistance. Reply for details.", # spam
    "WORK FROM HOME and earn $5,000 PER WEEK! No experience needed. Start today!", # spam
    "Your Netflix subscription has expired! Click here to renew and avoid service interruption.", # spam
    "URGENT SECURITY ALERT: Your email account has been compromised. Change your password immediately via this link.", # spam
    "YOU ARE A WINNER! Enter your details to claim your lottery jackpot prize of $1,000,000!", # spam
    "Get RICH Quick! Invest in our revolutionary new crypto currency. Guaranteed returns!", # spam
    "FINAL WARNING: Your domain name is about to expire! Renew now to keep your website online.", # spam
    "Cheap VIAGRA and CIALIS delivered discretely to your door. HUGE DISCOUNTS!", # spam
    "Mortgage Rates Are Plummeting! Refinance now and SAVE THOUSANDS! Free consultation.", # spam
    "Your parcel delivery was unsuccessful. Click here to reschedule and pay the re-delivery fee.", # spam
    "Don't miss out on this LIMITED TIME OFFER! 70 off designer watches. Shop now!", # spam
    "Improve your credit score INSTANTLY! Get approved for loans regardless of your credit history.", # spam
    "Congratulations! You've been selected for a government grant of $5,000. Apply now, no repayment required!", # spam
    "Are you still looking for hot singles in your area? Connect now for FREE!", # spam
    "Your Apple ID has been locked for security reasons. Verify your account through this link.", # spam
    "Get a FREE gift card just for taking our quick survey! Value up to $500!", # spam
    "UNSUBSCRIBE from future emails here: [malicious link]", # spam
    "Secret shopping jobs available! Earn up to $300 per assignment. Apply within.", # spam
    "Increase your 'manhood' by inches! Natural enlargement pills that work!", # spam
    "Your package is awaiting delivery. Confirm your shipping details to avoid return to sender.", # spam
    "Access exclusive content! Unlock premium features with a one-time payment. Click now!", # spam
    "Attention: Important message from the IRS regarding your tax refund. Urgent response required.", # spam
    "You have a new match on our dating site! View profile here: [dating site link].", # spam
    "Become a certified life coach in just 3 days! Enroll in our online course.", # spam
    "Debt consolidation solutions! Reduce your monthly payments and eliminate debt fast.", # spam
    "Your social security benefits are pending. Click to verify your eligibility and claim your funds.", # spam
    "Hot stocks to buy NOW! Get insider trading tips and make a fortune.", # spam
    "We detected unauthorized login attempts on your Google account. Secure your account now.", # spam
    "Get FREE stock market alerts! Never miss a winning trade again.", # spam
    "Limited-time offer! Your website is eligible for a top ranking on Google. Act fast!", # spam
    "New online casino with a HUGE welcome bonus! Play now and win big!", # spam
    "Your credit card has been compromised. Click to update your details and secure your account.", # spam
    "Earn money by testing new products from home! Join our paid survey panel.", # spam
    "Exclusive invitation: Attend our private investment seminar. Limited seats available.", # spam
    "Your package tracking information has been updated. View details and pay customs fees.", # spam
    "This is not a scam! Real opportunity to make passive income from home. Learn more.", # spam
    "Huge discounts on designer clothes! Shop now and save up to 80%.", # spam
    "Your car warranty is about to expire. Extend your coverage today to avoid costly repairs.", # spam
    "Free virus scan detected multiple threats! Click to clean your computer now.", # spam
    "Get a personal loan with no credit check! Fast approval and instant funds.", # spam
    "Protect your identity! Enroll in our credit monitoring service for free.", # spam
    "You've been pre-selected for a new government housing grant. Apply today!", # spam
    "Unlock your spiritual potential with our new meditation course. Special offer!", # spam
    "Last chance to claim your prize! Offer expires at midnight. Don't miss out!" # spam
]
for i, email_text in enumerate(new_emails):
    label, confidence = predict_spam_ham(email_text)
    print(f"البريد الإلكتروني {i+1}: '{email_text}'")
    print(f"  التصنيف: {label}, الثقة: {confidence:.4f}\n")

# --- 6. يمكنك الآن إدخال أي نص تريد اختباره ---
print("\n--- يمكنك الآن إدخال نصوص لاختبارها (اكتب 'exit' للخروج) ---")
while True:
    user_input = input("أدخل نص البريد الإلكتروني للاختبار: ")
    if user_input.lower() == 'exit':
        break
    
    label, confidence = predict_spam_ham(user_input)
    print(f"  التصنيف: {label}, الثقة: {confidence:.4f}\n")

print("انتهى البرنامج.")