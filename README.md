# Smart_IoT_Vanity

![mk](https://github.com/zecube/Smart_Makeup/assets/117415885/e7b6f282-9a64-47dd-ba57-589133da99b1)


The Smart IoT Vanity was a capstone design project that ran from October 2022 to October 2023. This innovative product features the following key functions:

✨ Real-time Makeup Experience

Experience various makeup looks virtually. Compare before and after looks to find makeup styles that suit your preferences.


✨ Optimized Online Cosmetics Shopping

The Smart IoT Vanity lets you preview various cosmetic products through virtual makeup try-ons. If you find products you love, you can purchase them online instantly. This innovative feature elevates cosmetics shopping to a smarter experience.


✨ IoT Furniture Control

The smart vanity syncs with surrounding IoT furniture like lights and speakers for convenient control.


✨ Digital Healthcare

- Melanoma Detection: Analyzes skin lesions to detect melanoma risk.
  
- Stress Relief: Recognizes facial expressions to assess stress levels and provide relief functions.
  
This product promises a smart lifestyle experience that goes beyond makeup and beauty to encompass health and wellbeing.

---

# Real-time Makeup Experience

![vid1_1](https://github.com/zecube/Smart_Makeup/assets/117415885/5789516a-46d4-4ee9-b7fb-4fd6ea0873ca)


First, we utilize the Mediapipe and dlib libraries to accurately recognize each part of the user's face. Then, using OpenCV, we separate individual layers like skin, lips, and eyes.

On these isolated layers, we apply scikit-learn's sophisticated color histogram matching technique to naturally transform them into the desired shades. Finally, we integrate the transformed layers to render a lifelike made-up look for the user. Most importantly, this entire process happens in real-time, providing an immersive experience akin to looking in a mirror.


![p11](https://github.com/zecube/Smart_Makeup/assets/117415885/f9871254-2c2f-46f0-bf01-7b48e28ed2b7)

To make the makeup experience even more personalized and intuitive, users can utilize the smart vanity's touch UI. With just a few simple steps, you can craft the perfect makeup look.

First, touch the area of the face you want to apply makeup to on the UI.

Second, tap through the various color options to find your ideal shade.

Third, easily adjust the opacity to control how intense or sheer you want the makeup application.

Finally, for features like blush or eyebrows, you can freely modify the default placement if desired.

This step-by-step process allows you to fully customize your perfect makeup according to your preferences. The touch interactions with the vanity UI make creating makeup looks fun and creative.

![2024-03-17 00 52 52](https://github.com/zecube/Smart_Makeup/assets/117415885/7d53231a-0df5-4448-95fe-f6858b4fe7c0)

---

# Online Cosmetics Shopping

![p13](https://github.com/zecube/Smart_Makeup/assets/117415885/35645719-f10f-4472-b4e8-945307b147f6)

After the virtual makeup trial, you simply need to add your desired colors to the cart. It's seamlessly integrated with a smartphone app for easy online purchasing.

This brings innovative practicality to the previously impractical online cosmetics market through virtual try-on experiences. By testing products beforehand, the likelihood of purchasing ill-fitting makeup that ends up wasted drastically decreases.

As a result, consumers enjoy a satisfying shopping experience, while businesses boost sales efficiency. Most importantly, from an environmental standpoint, cosmetic waste significantly reduces, increasing sustainability. The integration of virtual reality technology ushers transformative change across the cosmetics industry as a whole.

This technology was implemented through the integration of Firebase's real-time database between the Jetson board and Android platform.

---

# IoT Furniture Control & Digital Healthcare

![p14](https://github.com/zecube/Smart_Makeup/assets/117415885/b9fc16dc-e2f7-440b-9d2b-a716bdfacbfb)

Through the integration with Firebase's real-time database, we can control the Arduino-based air purifier we built. The sensor data from the air purifier can be monitored on the smart vanity and Android app, which also allows power control over the device.

On the smart vanity screen, you can check the air purifier status and control its power by tapping the IoT control button. The Android app provides an air purifier list view to access the related information.

---

![p1499](https://github.com/zecube/Smart_Makeup/assets/117415885/5846483a-1460-4216-b83c-1af4956d0e1e)

This vanity table incorporates digital healthcare features for melanoma detection and stress management. Users can easily activate the melanoma detection and stress relief through facial expression analysis functions by simply touching the button on the vanity table.

This vanity table utilizes the ABCDE technique for melanoma detection. It also leverages a Keras-based AI model to analyze the user's facial expressions. If negative expressions are detected, indicating a high risk of stress, it plays relaxing music to help alleviate the user's stress levels.

---

