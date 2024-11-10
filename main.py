
import openai
from openai import OpenAI
import base64
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import PIL.Image
import os
# from google.colab import userdata
from dotenv import load_dotenv
load_dotenv()

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_model(model_path):
    try:
        # Load the entire model
        model = tf.keras.models.load_model(model_path)
        return model
    except:
        # If loading fails, recreate the model architecture and load weights
        model = Sequential([
            Conv2D(512, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dropout(0.35),
            Dense(4, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        # Load weights
        model.load_weights(model_path)
        return model

  #section ai
# genai.configure(api_key= os.getenv('Google_API_KEY'))


def generate_explanation(model, model_prediction, confidence):

  prompt = f""" 
  You are an expert neurologist analyzing a brain tumor MRI scan with an overlaid saliency map. This map highlights in light cyan the areas the deep learning model focused on to classify the brain tumor. The model has classified the tumor as '{model_prediction}' with a confidence level of {confidence*100}%.

  In your analysis:
  - Identify the specific brain regions in the MRI that the model is concentrating on, based on the light cyan areas in the saliency map, and describe any significant anatomical landmarks or structures that fall within these highlighted regions.
  - Discuss plausible reasons why the model focused on these specific regions for the given classification, considering the typical tumor characteristics and growth patterns associated with glioma, meningioma, pituitary tumors, or no tumor.
  - Offer an insight into how the highlighted areas might correlate with the model's predicted tumor type.

  Do not mention the model at all in your explanation.
  Do not mention the work 'model' in your explanation.

  Keep your explanation concise and limited to 4 sentences. Ensure clarity and precision in your analysis by verifying each step logically.
    """

  # img = PIL.Image.open(img_path)
  image_path = os.path.join(output_dir, uploaded_file.name)
  with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

  # model = genai.GenerativeModel(model_name= "gemini-1.5-flash")
  # response = model.generate_context([prompt, img])

  # return response.text

  response = client.chat.completions.create(
        model="gpt-4o",
        # model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        # max_tokens=300
    )
    
  return response.choices[0].message.content

def generate_saliency_map(model, img_array, class_index, img_size):
  with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array)
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    # Resize gradients to match the original image size
    gradients = cv2.resize(gradients, img_size)

    # Create a circular mask for the brain area
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

    # Apply the mask to gradients
    gradients = gradients * mask

    # Normalize only the brain area
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
      brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    # Apply a higher threshold
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    # Apply more aggressive smoothing
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    # Create a heatmap overlay with enhanced contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, img_size)

    # Superimpose the heatmap on original image with increased opacity
    original_image = image.img_to_array(img)
    superimposed_img = heatmap * 0.7 + original_image * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    img_path = os.path.join(output_dir, uploaded_file.name)
    with open(img_path, 'wb') as f:
      f.write(uploaded_file.getbuffer())

    saliency_map_path = f"saliency_maps/{uploaded_file.name}"

    # Save the saliency map
    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img

def load_xception_model(model_path):
  img_shape = (299, 299, 3)
  base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape= img_shape, pooling='max')

  model = Sequential([base_model, Flatten(), Dropout(rate= 0.3), Dense(128, activation= 'relu'), Dropout(rate= 0.25), Dense(4, activation='softmax')])

  model.build((None,) + img_shape)

  # Compile the model
  model.compile(Adamax(learning_rate= 0.001), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

  model.load_weights(model_path)

  return model

st.title("Brain Tumor Classification")

# Upload Image
st.write("Upload an image of a brain MRI scan to classify the type of tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

  selected_model = st.radio(
    "Select Model",
    ("Transfer Learning - Xception", "Custom CNN")  
  )

  if selected_model == "Transfer Learning - Xception":
    model = load_xception_model("xception_model.weights.h5")
    img_size = (299, 299)
  else:
    model = load_model("cnm_model.weights.h5")
    img_size = (224, 224)

  labels = ['Glioma', 'Meningioma', 'No_tumor', 'Pituitary']
  img = image.load_img(uploaded_file, target_size=img_size)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0

  predictions = model.predict(img_array)

  # Get the class with the highest probability
  class_index = np.argmax(predictions[0])
  result = labels[class_index]

  st.write(f"Predicted Class: {result}")
  st.write("Predictions:")
  for label, prob in zip(labels, predictions[0]):
    st.write(f"{label}: {prob:.4f}")

  # Generate saliency map
  saliency_map = generate_saliency_map(model, img_array, class_index, img_size)

  # Save saliency map
  # saliency_map_path = f"saliency_maps/{uploaded_file.name}"
  # cv2.imwrite(saliency_map_path, cv2.cvtColor(saliency_map, cv2.COLOR_RGB2BGR))

  col1, col2 = st.columns(2)
  with col1:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
  with col2:
    st.image(saliency_map, caption="Saliency Map", use_container_width=True)

  st.write("## Classification Results:")

  result_container = st.container()
  result_container = st.container()
  result_container.markdown(
    f""" 
    <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1; text-align: center;">
          <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
          <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
            {result}
          </p>
        </div>
        <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
        <div style="flex: 1; text-align: center;">
          <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
          <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0">
            {predictions[0][class_index]:.4%}
          </p>
        </div>
        
      </div>
    </div>
    """,
    unsafe_allow_html= True
  )

  # Prepare data for Plotly chart
  probabilities = predictions[0]
  sorted_indices = np.argsort(probabilities)[::-1]
  sorted_labels = [labels[i] for i in sorted_indices]
  sorted_probabilities = probabilities[sorted_indices]

  # Create a Plotly chart
  fig = go.Figure(go.Bar(
  x=sorted_probabilities,
  y=sorted_labels,
  orientation= 'h',
  marker_color=['red' if label == result else 'blue' for label in sorted_labels]
  ))

  # Customize the chart layout
  fig.update_layout(
  title='Probabilities for each class',
  xaxis_title='Probability',
  yaxis_title='Class',
  height= 400,
  width=600,
  yaxis=dict(autorange= 'reversed')
  )

  # Add value labels to the bars
  for i, prob in enumerate(sorted_probabilities):
    fig.add_annotation(
      x=prob,
      y=i,
      text=f'{prob:.4f}',
      showarrow= False,
      xanchor='left',
      xshift=5
    )

  # Display the Plotly chart
  st.plotly_chart(fig)

  saliency_map_path = f'saliency_maps/{uploaded_file.name}'

  explanation = generate_explanation(saliency_map_path, result, predictions[0][class_index])

  st.write("## Explanation:")
  st.write(explanation)

