import streamlit as st
import cv2
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Set page title and favicon
st.set_page_config(
    page_title="Erythrocyte Analysis App",
    page_icon="🔬"
)

def get_erythrocyte_shape_factors(image, anomaly_threshold=1.7):
    """
    Analyzes an image of erythrocytes and returns processed data.
    Erythrocytes are colored based on their shape factor.
    """
    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors_data = []
    anomalies_data = []

    for contour in contours:
        if len(contour) > 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                minor_axis = min(axes)
                major_axis = max(axes)
                
                if minor_axis > 0:
                    shape_factor = major_axis / minor_axis
                    
                    # Determine color based on shape factor
                    ellipse_color = (0, 255, 0) # Green for normal (SF <= 1.3)
                    if shape_factor > 1.3 and shape_factor <= 1.5:
                        ellipse_color = (0, 255, 255) # Yellow for moderately elongated
                    elif shape_factor > 1.5:
                        ellipse_color = (0, 0, 255) # Red for highly elongated

                    if shape_factor <= anomaly_threshold:
                        shape_factors_data.append({
                            "Erythrocyte Number": len(shape_factors_data) + 1,
                            "Shape Factor": shape_factor
                        })
                        cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                        angle_rad = np.radians(orientation)
                        major_end_point_1 = (int(center[0] + major_axis/2 * np.cos(angle_rad)),
                                            int(center[1] + major_axis/2 * np.sin(angle_rad)))
                        major_end_point_2 = (int(center[0] - major_axis/2 * np.cos(angle_rad)),
                                            int(center[1] - major_axis/2 * np.sin(angle_rad)))
                        cv2.line(processed_image, major_end_point_1, major_end_point_2, (0, 0, 255), 1)
                        minor_angle_rad = np.radians(orientation + 90)
                        minor_end_point_1 = (int(center[0] + minor_axis/2 * np.cos(minor_angle_rad)),
                                            int(center[1] + minor_axis/2 * np.sin(minor_angle_rad)))
                        minor_end_point_2 = (int(center[0] - minor_axis/2 * np.cos(minor_angle_rad)),
                                            int(center[1] - minor_axis/2 * np.sin(minor_angle_rad)))
                        cv2.line(processed_image, minor_end_point_1, minor_end_point_2, (255, 0, 0), 1)
                        number_text = str(len(shape_factors_data))
                        text_pos = (int(center[0]) + 15, int(center[1]))
                        cv2.putText(processed_image, number_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        anomalies_data.append({
                            "Erythrocyte Number": len(anomalies_data) + 1,
                            "Shape Factor": shape_factor
                        })
                        cv2.ellipse(processed_image, ellipse, (255, 0, 255), 2)
                        number_text = f"A{len(anomalies_data)}"
                        text_pos = (int(center[0]) + 15, int(center[1]))
                        cv2.putText(processed_image, number_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except cv2.error:
                continue

    return processed_image, shape_factors_data, anomalies_data

# --- Streamlit UI ---
st.title("🔬 Erythrocyte Analysis App")
st.markdown("### Upload an image for analysis or use the default example")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
use_default_image = st.checkbox("Use default image (A.jpg)", value=True)

# Sidebar controls
st.sidebar.header("Analysis Settings")
anomaly_threshold_slider = st.sidebar.slider(
    "Anomaly detection threshold (Shape Factor >)", 
    min_value=1.5, 
    max_value=2.5, 
    value=1.7, 
    step=0.05
)

run_button = st.button("Run Analysis")

if run_button:
    image_to_process = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.success("Image uploaded successfully!")

    elif use_default_image:
        default_url = "https://raw.githubusercontent.com/slastrzelec/erytrocyty-analyze/main/A.jpg"
        try:
            response = requests.get(default_url)
            response.raise_for_status()
            image_array = np.frombuffer(response.content, np.uint8)
            image_to_process = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            st.info("Default image used.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching default image: {e}")
            image_to_process = None

    if image_to_process is not None:
        processed_img, shape_factors, anomalies = get_erythrocyte_shape_factors(image_to_process, anomaly_threshold_slider)

        st.subheader("Analysis Results")
        st.image(processed_img, channels="BGR", caption="Processed Erythrocytes")

        # Combine normal and anomaly data for plotting
        df_normal = pd.DataFrame(shape_factors)
        df_anomalies = pd.DataFrame(anomalies)

        if not df_normal.empty or not df_anomalies.empty:
            # --- Descriptive Statistics for Normal Cells ---
            if not df_normal.empty:
                avg_shape_factor = df_normal['Shape Factor'].mean()
                median_shape_factor = df_normal['Shape Factor'].median()
                std_dev_shape_factor = df_normal['Shape Factor'].std()
                min_shape_factor = df_normal['Shape Factor'].min()
                max_shape_factor = df_normal['Shape Factor'].max()
                
                st.markdown("---")
                st.subheader("Descriptive Statistics for Normal Cells")
                st.info(
                    f"**Average Shape Factor**: {avg_shape_factor:.2f}\n\n"
                    f"**Median**: {median_shape_factor:.2f}\n\n"
                    f"**Standard Deviation**: {std_dev_shape_factor:.2f}\n\n"
                    f"**Minimum**: {min_shape_factor:.2f} | **Maximum**: {max_shape_factor:.2f}"
                )

            # --- Distribution Plots with Anomalies ---
            st.markdown("---")
            st.subheader("Shape Factor Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))

            if not df_normal.empty:
                ax.hist(df_normal['Shape Factor'], bins=15, alpha=0.6, color='b', edgecolor='black', label='Normal')
            if not df_anomalies.empty:
                ax.hist(df_anomalies['Shape Factor'], bins=15, alpha=0.6, color='m', edgecolor='black', label='Anomaly')

            ax.set_title('Erythrocyte Shape Factor Distribution (Histogram)')
            ax.set_xlabel('Shape Factor')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            # --- Boxplot ---
            st.markdown("---")
            st.subheader("Boxplot of Shape Factors")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            data_to_plot = []
            labels = []
            if not df_normal.empty:
                data_to_plot.append(df_normal['Shape Factor'])
                labels.append('Normal')
            if not df_anomalies.empty:
                data_to_plot.append(df_anomalies['Shape Factor'])
                labels.append('Anomaly')

            ax2.boxplot(data_to_plot, vert=False, patch_artist=True, labels=labels,
                        boxprops=dict(facecolor='blue', color='blue'),
                        medianprops=dict(color='black'))
            if not df_anomalies.empty:
                ax2.boxplots = ax2.boxplot([df_normal['Shape Factor'], df_anomalies['Shape Factor']], vert=False, patch_artist=True)
            ax2.set_xlabel('Shape Factor')
            ax2.grid(True)
            st.pyplot(fig2)

            # --- Results Table ---
            st.markdown("---")
            st.subheader("Results Table")
            st.dataframe(pd.concat([df_normal, df_anomalies], ignore_index=True))

        else:
            st.warning("No erythrocytes detected.")

    elif uploaded_file is None and not use_default_image:
        st.warning("Please upload a file or select the default image option.")
