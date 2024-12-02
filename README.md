# Intelligent Customer Interaction System Using Facial Recognition

Welcome to the Intelligent Customer Interaction System Using Facial Recognition project! This project aims to enhance the shopping experience by utilizing facial recognition technology to provide personalized product recommendations and email notifications.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project leverages computer vision and machine learning to recognize customers' faces and provide them with tailored product recommendations based on their previous purchases and seasonal preferences. The system also sends email notifications for requested purchases, making it a seamless and engaging shopping experience.

## Features

- **Facial Recognition**: Detect and recognize customers using a webcam.
- **Personalized Recommendations**: Suggest items based on previous purchases and seasonal preferences.
- **Email Notifications**: Automatically send emails for requested purchases.
- **Interactive UI**: User-friendly interface for easy interaction and selection of products.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- SMTP library
- Haar Cascade XML file for face detection

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download the Haar Cascade XML file**:
    Download `haarcascade_frontalface_default.xml` from the OpenCV GitHub repository and place it in the project directory.

4. **Prepare the datasets**:
    - Create a `datasets` directory and subdirectories for each customer.
    - Place face images of each customer in their respective subdirectory.

5. **Prepare item images**:
    - Create an `items` directory and place images of items to be recommended.

## Usage

1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Interact with the system**:
    - Ensure your webcam is functioning properly.
    - The system will recognize your face and display personalized recommendations.
    - Click on the categories and items to request purchases.

## Configuration

- **Email Settings**:
    - Update the `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, and `EMAIL_TO` variables in the script to configure the email notifications.

- **CSV File for Recommendations**:
    - Ensure `recommendations.csv` is formatted correctly and placed in the project directory.
    - The CSV file should have columns: `Customer Name`, `Previous Purchases`, `Seasonal Items (Winter)`, and `Seasonal Items (Summer)`.

## Contributing

We welcome contributions to enhance the functionality and features of this project! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Thank you for checking out this project! If you have any questions or need further assistance, feel free to open an issue or contact the project maintainers.
