import streamlit as st
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# Load the model
model=tf.keras.models.load_model('Denoising_image_HSI.keras')

# Function to process and denoise the image
def denoise_image(uploaded_file):
    new_data = sio.loadmat(uploaded_file)
    new_img_noisy_np = new_data["image_noisy"].astype(np.float32)
    new_img_noisy = new_img_noisy_np.reshape(1, new_img_noisy_np.shape[0], new_img_noisy_np.shape[1], new_img_noisy_np.shape[2])

    # Predict the denoised image
    denoised_image = model.predict(new_img_noisy)
    return new_img_noisy_np, denoised_image[0]

# Streamlit interface
st.title("HSI Image Denoising")
st.write("Upload a .mat file containing the noisy image.")

uploaded_file = st.file_uploader("Choose a .mat file", type="mat")

if uploaded_file is not None:
    # Process the uploaded file and get the denoised image
    original_image, denoised_image = denoise_image(uploaded_file)

    band = [57, 27, 17]  # Example bands to display

    # Display the original image
    st.subheader("Original Image")
    fig, ax = plt.subplots()
    ax.imshow(original_image[:, :, band])
    ax.axis("off")
    st.pyplot(fig)

    # Display the denoised image
    st.subheader("Denoised Image")
    fig, ax = plt.subplots()
    ax.imshow(denoised_image[:, :, band])
    ax.axis("off")
    st.pyplot(fig)
