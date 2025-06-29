# Multi-Task Siamese Network for Face Verification and Gender Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project by **Team Quixotic** that implements a multi-task Siamese network to simultaneously perform face verification (are two images the same person?) and gender classification. The model leverages a pre-trained FaceNet backbone and a two-stage fine-tuning process to achieve high accuracy on both tasks.

---

### Key Features

-   **Dual-Task Learning:** Performs face verification and gender classification in a single, efficient forward pass.
-   **High Accuracy:** Achieves strong performance by fine-tuning a state-of-the-art FaceNet (InceptionResNetV1) model.
-   **Robust Preprocessing:** Utilizes an MTCNN face detector with padding and a center-crop fallback to handle a wide variety of input images.
-   **Class Imbalance Handling:** Employs a weighted loss function to effectively train on a dataset with significant gender imbalance.
-   **Advanced Training Strategy:** Uses a two-stage training regimen (head-only followed by full-model fine-tuning) to prevent catastrophic forgetting.
-   **Interactive Dashboard:** Comes with a comprehensive testing dashboard built with `ipywidgets` to test the model on single pairs or entire datasets.

---

## Table of Contents
1.  [Technical Methodology](#technical-methodology)
2.  [Results](#results)
3.  [Setup and Installation](#setup-and-installation)
4.  [How to Use](#how-to-use)
5.  [Acknowledgements](#acknowledgements)

---

## Technical Methodology

The model's architecture and training process were designed to be both robust and effective.

### Siamese Network Architecture
We employ a Siamese network, which uses a shared backbone to process two input images and generate 512-dimensional feature embeddings.

-   **Backbone:** A pre-trained **FaceNet** model (InceptionResNetV1) serves as the feature extractor. Its weights are initially frozen to leverage its powerful, generalized knowledge of facial features.
-   **Verification Head:** A `Lambda` layer calculates the Euclidean distance between the two embeddings. A smaller distance implies a higher probability that the images are of the same person.
-   **Gender Classification Head:** A shared `Dense` layer with `Dropout` and L2 regularization predicts the gender from each embedding.

### Two-Stage Training Regimen
To adapt the model without destroying its pre-trained weights, we use a two-stage approach:

1.  **Head-Only Training:** First, we freeze the entire FaceNet backbone and train only the newly added verification and gender heads. This allows them to learn the specifics of our dataset quickly.
2.  **Full-Model Fine-Tuning:** Next, we unfreeze the top layers of the FaceNet backbone and continue training the entire model with a very low learning rate (`AdamW` optimizer with `CosineDecay`). This allows the entire network to make subtle, cohesive adjustments.

 
![A simplified diagram illustrating the multi-task approach](Comsys-Hackathon-original-by-me-team-Quixotic/img/diag.jpg)

---

## Results

The model's performance was evaluated on a held-out validation set. A comprehensive dashboard was created to visualize key metrics for both tasks.

-   **Face Verification:** The model shows excellent separation between "match" and "no-match" distance distributions, achieving high accuracy and a strong AUC score.
-   **Gender Classification:** The model performs well, though performance on the minority class (female) highlights the challenge of data imbalance, a key area for future work.


![Evaluation Dashboard Screenshot](Comsys-Hackathon-original-by-me-team-Quixotic/img/output1.png)

---

## Setup and Installation

To run this project locally, please follow these steps.

**Prerequisites:**
-   Python 3.9+
-   `pip` and `venv`
-   Git

**1. Clone the Repository**
```
git clone https://github.com/MasterKraid/Comsys-Hackathon-original-by-me-team-Quixotic.git
cd your-repository-name
```

**2. Create a Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.

```
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

```

**3. Install Dependencies**
This project relies on several packages, including `tensorflow`, `keras-facenet`, and `mtcnn`. Install them using the provided `requirements.txt` file.

*(If you haven't created one, run `pip freeze > requirements.txt` in your activated environment after installing all packages.)*
```bash
pip install -r requirements.txt
```
**Note:** This project is best run on a machine with a **GPU** and CUDA installed, as CPU-based training will be extremely slow. The notebook is pre-configured for environments like Kaggle or Google Colab.

---

## How to Use

**1. Data Structure**
The notebook expects a specific data structure. Place your datasets in a root folder and point the path variables in **Cell 2** accordingly.

```
/path/to/your/data/
├── Task_A/                 # Used to create the gender map
│   ├── train/
│   │   ├── male/
│   │   └── female/
│   └── val/
│       ├── male/
│       └── female/
└── Task_B/                 # Primary dataset for training
    └── train/
        ├── Person_1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── Person_2/
            ├── imgA.jpg
            └── ...
```

**2. Running the Notebook**
Open the `comsys-hackathon-test.ipynb` notebook and execute the cells in order.

-   **Data Caching (Cell 4.5):** The first run will be very slow as it detects, crops, and caches every face. Subsequent runs will be much faster as they will load directly from the cache.
-   **Training (Cells 8 & 9):** The model will undergo the two-stage training process.
-   **Evaluation (Cell 12):** The final, comprehensive evaluation dashboard will be generated.
-   **Interactive Testing (Cell 14):** Use the interactive widgets to test the trained model on your own images or test datasets.

---

## Acknowledgements

This project was brought to life through dedicated teamwork, collaboration, and support.

### Core Development Team
This model was designed, implemented, and trained by **Team Quixotic**.

-   **Sohan Das** - *Team Lead*
    -   Responsible for project management, strategic direction, and overall coordination.
-   **Tathagata (Kraid) S.** - *Lead Developer*
    -   Responsible for the end-to-end implementation of the data pipeline, model architecture, training regimen, and evaluation framework.

### Special Regards
We extend a special thank you to Mahiyan (**Aj**), who provided invaluable assistance by creating a custom testing dataset used for the final interactive dashboard. This contribution was crucial for robustly validating the model's real-world performance.

---
