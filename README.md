# MNIST — Neural Network From Scratch

A handwritten digit recognition project built by implementing a **Neural Network from scratch using only NumPy**.

The project has two main parts:

- A NumPy-based neural network trained on the MNIST dataset.
- An interactive Pygame application for drawing a digit and getting the model's prediction.

## Neural Network

The neural network was implemented from scratch using **NumPy**, without using deep-learning frameworks such as PyTorch or TensorFlow.

The model was trained on the **MNIST handwritten digit dataset** and achieved a test accuracy of:

**97.44%**

The complete neural network implementation, including the forward pass, backpropagation, and parameter updates, can be found in ```window/NN.py```

The training and experimentation notebook is ```notebook/neuralnetwork.ipynb```

## Interactive Digit Recognizer

The trained model is connected to a Pygame interface where you can draw a handwritten digit using your mouse.

The model processes the drawing and predicts which digit was drawn, along with the probability of its prediction.

The Pygame implementation is located in ```window/draw_window.py```

To launch the interactive application : ```python main.py```

## Project Structure

MNIST/
├── notebook/
│   └── neuralnetwork.ipynb
├── window/
│   ├── NN.py
│   └── draw_window.py
├── main.py
├── weights_nn.npz
├── README.md
└── pyproject.toml