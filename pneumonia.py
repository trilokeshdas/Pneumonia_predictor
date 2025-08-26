import os
import streamlit as st
import tensorflow as tf
import gdown

st.title('X-Ray Image Classifier')

IMG_SIZE = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# Google Drive file ID (replace with your own model file's ID)
FILE_ID = "16-u_JQ-b3Xl0mgDVeME3_Vdk0rWHlpIa"  
OUTPUT_PATH = "custom_pre_trained_model_10.h5"

# Download model from Google Drive if not already present
if not os.path.exists(OUTPUT_PATH):
    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, OUTPUT_PATH, quiet=False)
        st.write("✅ Model downloaded from Google Drive")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load model
try:
    model = tf.keras.models.load_model(OUTPUT_PATH)
    print("Model Loaded")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

def predict_image(file):
    img = tf.keras.preprocessing.image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0

    prediction = model.predict(img_array)
    prob_pneumonia = float(prediction[0][0])
    prob_normal = 1 - prob_pneumonia
    pred_class = 1 if prob_pneumonia >= 0.5 else 0
    confidence = prob_pneumonia if pred_class == 1 else prob_normal

    return pred_class, confidence, prob_normal, prob_pneumonia

def patient_status(prob_pneumonia, prob_normal):
    if prob_normal >= 0.9:
        return "**The patient is likely healthy. No signs of Pneumonia detected.**"
    elif prob_pneumonia < 0.1:
        return "**Very low pneumonia risk. The patient appears normal.**"
    elif prob_pneumonia < 0.4:
        return "**Low risk of Pneumonia. Keep monitoring if symptoms appear.**"
    elif prob_pneumonia < 0.7:
        return "**Moderate risk. Further medical tests are recommended.**"
    else:
        return "**High likelihood of Pneumonia. Consult a doctor immediately.**"

def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader("Choose an X-Ray image", type=['jpeg', 'jpg', 'png'])

    if file is not None:
        st.image(file, caption="Uploaded X-Ray", use_container_width=True)

        if st.button("PREDICT"):
            with st.spinner("Analyzing X-Ray... Please wait..."):
                pred_class, pred_conf, prob_normal, prob_pneumonia = predict_image(file)

            st.markdown(f"### **Prediction: {CATEGORIES[pred_class]}**")
            st.write(f"Confidence: **{round(pred_conf * 100, 2)}%**")
            st.write(f"Normal: {round(prob_normal * 100, 2)}%")
            st.write(f"Pneumonia: {round(prob_pneumonia * 100, 2)}%")

            # Patient status conclusion
            st.markdown("---")
            st.markdown(patient_status(prob_pneumonia, prob_normal))

def main():
    load_classifier()

if __name__ == "__main__":
    main()
