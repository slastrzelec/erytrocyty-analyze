import cv2
import numpy as np
import requests

def get_erythrocyte_shape_factors(image_url):
    """
    Downloads an image from a URL, analyzes the erythrocytes, and returns a list
    of their shape factors without any visual output.
    """
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check for successful response

        # Convert byte data to a NumPy array and decode it into an OpenCV image
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print("Could not decode the image from the URL.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the image: {e}")
        return []

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate the cells from the background
    # You may need to adjust the threshold value (100) for different images
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the erythrocytes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors = []

    for contour in contours:
        # Filter out small contours which are likely noise
        if len(contour) > 5:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse

            # The axes are returned as (minor_axis, major_axis)
            minor_axis = min(axes)
            major_axis = max(axes)

            # Calculate the shape factor (ratio of major to minor axis)
            if minor_axis > 0:
                shape_factor = major_axis / minor_axis
                shape_factors.append(shape_factor)

    return shape_factors

# The URL of the image
IMAGE_URL = "https://raw.githubusercontent.com/slastrzelec/erytrocyty-analyze/main/A.jpg"

# Call the function and print the results
shape_factors = get_erythrocyte_shape_factors(IMAGE_URL)

if shape_factors:
    print(f"Found {len(shape_factors)} erythrocytes.")

    # Print individual shape factors
    for i, factor in enumerate(shape_factors):
        print(f"Erythrocyte {i+1}: {factor:.2f}")

    # Calculate and print the average shape factor
    avg_shape_factor = np.mean(shape_factors)
    print(f"\nAverage shape factor: {avg_shape_factor:.2f}")
else:
    print("No erythrocytes were found for analysis.")