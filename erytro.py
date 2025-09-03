import cv2
import numpy as np
import requests
import pandas as pd
import os

def get_erythrocyte_shape_factors(image_url, excel_filename="shape_summary.xlsx"):
    """
    Downloads an image, analyzes erythrocytes, displays processed image,
    and opens an Excel file with shape factor summary.
    """
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

    shape_factors = []

    for contour in contours:
        if len(contour) > 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            center = (int(center[0]), int(center[1]))
            minor_axis = min(axes)
            major_axis = max(axes)

            if minor_axis > 0:
                shape_factor = major_axis / minor_axis
                if shape_factor <= 1.7:
                    shape_factors.append(shape_factor)
                    cv2.ellipse(processed_image, ellipse, (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Processed Erythrocytes", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create Excel summary and open it
    if shape_factors:
        df = pd.DataFrame({
            "Erythrocyte": [f"Erythrocyte {i+1}" for i in range(len(shape_factors))],
            "Shape Factor": shape_factors
        })
        df.loc["Average"] = ["Average", np.mean(shape_factors)]
        # Save Excel temporarily
        df.to_excel(excel_filename, index=False)
        # Open Excel
        os.startfile(excel_filename)
        print(f"Excel summary opened: {excel_filename}")

    return shape_factors

# URL of the image
IMAGE_URL = "https://raw.githubusercontent.com/slastrzelec/erytrocyty-analyze/main/A.jpg"

# Run analysis
shape_factors = get_erythrocyte_shape_factors(IMAGE_URL)
if shape_factors:
    print(f"Found {len(shape_factors)} erythrocytes.")
    for i, factor in enumerate(shape_factors):
        print(f"Erythrocyte {i+1}: {factor:.2f}")
    print(f"\nAverage shape factor: {np.mean(shape_factors):.2f}")
else:
    print("No erythrocytes were found for analysis.")
