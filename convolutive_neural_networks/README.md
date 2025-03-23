# 🧠 CNN Convolutional Neural Networks

## 📁 Project Structure

```
artificial_vision/
│
├── .gitignore
├── README.md                  # Root project README
├── requirements.txt           # Global project dependencies
│
├── convolutive_neural_networks/
│   ├── README.md              # Module-specific README
│   ├── main.py                # Entry point for CNN operations
│   └── src/
│       ├── managing_images.py        # Image management module
│       └── convolution_kernels.py    # Convolution kernel logic
│   │
│   └── documentation/
│       ├── cnn_architecture  # CNN architecture explanation
│       └── cnn_layers.py     # CNN layers details
│
└── neural_networks/
    ├── activations_functions/
    ├── keras_networks/
    └── numpy_networks/
    
```

---

## ⚙️ Setup Instructions

Follow the steps below to set up and run the project.

### ✅ Step 1: Clone the Repository

```bash
git clone git@github.com:CharlyMercury/artificial_vision.git
cd artificial_vision
```

### ✅ Step 2: Create a Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### ✅ Step 3: Install Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

---


## 🚀 How to Run the Project

Go to folder where is located the `main.py` of this project:

```bash
cd convolutive_neural_networks
```


The `main.py` script accepts one of three arguments:

```bash
python main.py <option>
```

### Available options:

| Command         | Description                             |
|----------------|-----------------------------------------|
| `mng_imgs`      | Run the image management module         |
| `conv_ker`      | Run the convolution kernel processor    |
| `conv_video`    | Apply a selected kernel to video/image  |

### Example usage:

```bash
python main.py mng_imgs
python main.py conv_ker
python main.py conv_video
```

If no valid argument is given, the script will show usage instructions.

---
