# Deep Learning Lab Experiments (Weeks 03–10)

Short Python implementations based on the experiments visible in the uploaded syllabus image.

## Setup

Python 3.10+ is recommended.

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

Some experiments use `tensorflow-datasets`:

```bash
pip install tensorflow-datasets
```

Run a script from this folder, for example:

```bash
python week03_gates.py
```

## Experiments included

- Week 03: FNN for AND, OR and XOR gates.
- Week 04: Character recognition using CNN and RNN (MNIST digits are used as a compact character-recognition example).
- Week 05: Image captioning with CNN + LSTM. Requires the Flickr8k dataset and is a compact training template.
- Week 06: MNIST autoencoder and GAN.
- Week 07: CNN-based image filtering / edge detection on MNIST.
- Week 08: Autoencoder feature compression followed by K-Means clustering.
- Week 09: Transfer learning with MobileNetV2 on CIFAR-10.
- Week 10: Mini-project example: image classification pipeline with training, evaluation and prediction.

## Dataset notes

MNIST and CIFAR-10 are downloaded automatically by Keras. The Week 05 script expects:
- `Flickr8k_Dataset/` containing image files
- `Flickr8k.token.txt` containing lines in the usual `image.jpg#0<TAB>caption` format

The scripts are intentionally kept short for lab use. Training quality depends on epochs, hardware and dataset size.
