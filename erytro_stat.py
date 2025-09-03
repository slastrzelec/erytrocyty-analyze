import cv2
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt

def get_erythrocyte_shape_factors(image_url):
    """
    Downloads and analyzes an image of erythrocytes, calculating their shape factors.
    Displays processed image and plots (histogram + boxplot) with descriptive statistics.
    Erythrocytes are colored based on shape factor:
    - Green: normal (SF <= 1.3)
    - Yellow: moderately elongated (1.3 < SF <= 1.5)
    - Red: highly elongated (1.5 < SF <= 1.7)
    - Magenta: potential anomalies (SF > 1.7)
    """
    # --- Section 1: Download and preprocess image ---
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            print("Could not decode the image from the URL.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return []

    processed_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Section 2: Analyze shape factors ---
    shape_factors_data = []
    anomalies_data = []

    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            minor_axis = min(axes)
            major_axis = max(axes)
            if minor_axis > 0:
                shape_factor = major_axis / minor_axis

                # Determine color
                ellipse_color = (0, 255, 0)  # Green
                if 1.3 < shape_factor <= 1.5:
                    ellipse_color = (0, 255, 255)  # Yellow
                elif 1.5 < shape_factor <= 1.7:
                    ellipse_color = (0, 0, 255)  # Red

                if shape_factor <= 1.7:
                    shape_factors_data.append({
                        "Erythrocyte Number": len(shape_factors_data)+1,
                        "Shape Factor": shape_factor
                    })
                    # Draw ellipse and axes
                    cv2.ellipse(processed_image, ellipse, ellipse_color, 2)
                    angle_rad = np.radians(orientation)
                    major_end_1 = (int(center[0] + major_axis/2 * np.cos(angle_rad)),
                                   int(center[1] + major_axis/2 * np.sin(angle_rad)))
                    major_end_2 = (int(center[0] - major_axis/2 * np.cos(angle_rad)),
                                   int(center[1] - major_axis/2 * np.sin(angle_rad)))
                    cv2.line(processed_image, major_end_1, major_end_2, (0,0,255), 1)
                    minor_angle_rad = np.radians(orientation+90)
                    minor_end_1 = (int(center[0] + minor_axis/2 * np.cos(minor_angle_rad)),
                                   int(center[1] + minor_axis/2 * np.sin(minor_angle_rad)))
                    minor_end_2 = (int(center[0] - minor_axis/2 * np.cos(minor_angle_rad)),
                                   int(center[1] - minor_axis/2 * np.sin(minor_angle_rad)))
                    cv2.line(processed_image, minor_end_1, minor_end_2, (255,0,0), 1)
                    # Number
                    cv2.putText(processed_image, str(len(shape_factors_data)),
                                (int(center[0])+15, int(center[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    anomalies_data.append({
                        "Erythrocyte Number": len(anomalies_data)+1,
                        "Shape Factor": shape_factor
                    })
                    cv2.ellipse(processed_image, ellipse, (255,0,255), 2)
                    cv2.putText(processed_image, f"A{len(anomalies_data)}",
                                (int(center[0])+15, int(center[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    # --- Section 3: Display image ---
    cv2.imshow("Processed Erythrocytes", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Section 4: Display plots ---
    if shape_factors_data:
        df = pd.DataFrame(shape_factors_data)
        fig, axes = plt.subplots(1,2, figsize=(15,6))
        # Histogram
        axes[0].hist(df['Shape Factor'], bins=15, alpha=0.6, color='b', edgecolor='black')
        df['Shape Factor'].plot(kind='kde', color='r', ax=axes[0])
        axes[0].set_title("Erythrocyte Shape Factor Distribution (Histogram)")
        axes[0].set_xlabel("Shape Factor")
        axes[0].set_ylabel("Frequency / Density")
        axes[0].grid(True)
        # Boxplot
        axes[1].boxplot(df['Shape Factor'], vert=False, patch_artist=True)
        axes[1].set_title("Erythrocyte Shape Factor (Boxplot)")
        axes[1].set_xlabel("Shape Factor")
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()

        # Descriptive stats
        print(f"\nAverage SF: {df['Shape Factor'].mean():.2f}")
        print(f"Median SF: {df['Shape Factor'].median():.2f}")
        print(f"Std dev SF: {df['Shape Factor'].std():.2f}")

    if anomalies_data:
        print(f"\nFound {len(anomalies_data)} potential anomalies (SF > 1.7)")

    return [d['Shape Factor'] for d in shape_factors_data]

# --- Example usage ---
IMAGE_URL = "https://raw.githubusercontent.com/slastrzelec/erytrocyty-analyze/main/A.jpg"
get_erythrocyte_shape_factors(IMAGE_URL)
