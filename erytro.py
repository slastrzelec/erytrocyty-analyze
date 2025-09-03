# niech w excelu pokjawia się wsp
import cv2
import numpy as np
import requests

def get_erythrocyte_shape_factors(image_url):
    """
    Downloads an image from a URL, analyzes the erythrocytes, and returns a list
    of their shape factors. It also displays the processed image with numbered cells,
    their major/minor axes, and ellipses.
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

    # Create a copy of the original image to draw on
    processed_image = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to separate the cells from the background
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the erythrocytes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_factors = []

    # Iterate through contours with an index using enumerate
    for i, contour in enumerate(contours):
        # Filter out small contours which are likely noise
        if len(contour) > 5:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            center = (int(center[0]), int(center[1]))

            # The axes are returned as (minor_axis, major_axis)
            minor_axis = min(axes)
            major_axis = max(axes)

            # Calculate the shape factor (ratio of major to minor axis)
            if minor_axis > 0:
                shape_factor = major_axis / minor_axis
                # Add condition to reject shape factors greater than 1.7
                if shape_factor <= 1.7:
                    shape_factors.append(shape_factor)

                    # Draw the ellipse on the processed image
                    cv2.ellipse(processed_image, ellipse, (0, 255, 0), 2)

                    # Calculate and draw the major and minor axes
                    angle_rad = np.radians(orientation)
                    
                    # Major axis (red line)
                    major_end_point_1 = (int(center[0] + major_axis/2 * np.cos(angle_rad)),
                                        int(center[1] + major_axis/2 * np.sin(angle_rad)))
                    major_end_point_2 = (int(center[0] - major_axis/2 * np.cos(angle_rad)),
                                        int(center[1] - major_axis/2 * np.sin(angle_rad)))
                    cv2.line(processed_image, major_end_point_1, major_end_point_2, (0, 0, 255), 1)

                    # Minor axis (blue line)
                    minor_angle_rad = np.radians(orientation + 90)
                    minor_end_point_1 = (int(center[0] + minor_axis/2 * np.cos(minor_angle_rad)),
                                        int(center[1] + minor_axis/2 * np.sin(minor_angle_rad)))
                    minor_end_point_2 = (int(center[0] - minor_axis/2 * np.cos(minor_angle_rad)),
                                        int(center[1] - minor_axis/2 * np.sin(minor_angle_rad)))
                    cv2.line(processed_image, minor_end_point_1, minor_end_point_2, (255, 0, 0), 1)

                    # Display the erythrocyte number
                    number_text = str(len(shape_factors))
                    text_pos = (int(center[0]) + 15, int(center[1]))
                    cv2.putText(
                        processed_image, 
                        number_text, 
                        text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
                    
    # Display the processed image
    cv2.imshow("Processed Erythrocytes", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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