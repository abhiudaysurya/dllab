# Compact CNN + LSTM image-captioning training template.
# Dataset required: Flickr8k_Dataset/ and Flickr8k.token.txt
import os, numpy as np, tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Model

IMAGE_DIR, CAP_FILE = "Flickr8k_Dataset", "Flickr8k.token.txt"
captions = {}
with open(CAP_FILE, encoding="utf-8") as f:
    for line in f:
        name, cap = line.strip().split("\t")
        captions.setdefault(name.split("#")[0], []).append("startseq " + cap.lower() + " endseq")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sum(captions.values(), []))
vocab = len(tokenizer.word_index) + 1
max_len = max(len(c.split()) for cs in captions.values() for c in cs)

# Pre-trained CNN extracts a feature vector for each image.
cnn = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
def image_feature(path):
    img = tf.keras.utils.load_img(path, target_size=(299, 299))
    x = tf.keras.utils.img_to_array(img)[None, ...]
    return cnn.predict(preprocess_input(x), verbose=0)[0]

Ximg, Xseq, Y = [], [], []
for name, caps in captions.items():
    path = os.path.join(IMAGE_DIR, name)
    if not os.path.exists(path): continue
    feat = image_feature(path)
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            Ximg.append(feat)
            Xseq.append(pad_sequences([seq[:i]], maxlen=max_len)[0])
            Y.append(tf.keras.utils.to_categorical(seq[i], vocab))

Ximg, Xseq, Y = np.array(Ximg), np.array(Xseq), np.array(Y)
image_in = layers.Input((2048,))
text_in = layers.Input((max_len,))
a = layers.Dense(128, activation="relu")(image_in)
b = layers.Embedding(vocab, 128, mask_zero=True)(text_in)
b = layers.LSTM(128)(b)
out = layers.Dense(vocab, activation="softmax")(layers.Add()([a, b]))
model = Model([image_in, text_in], out)
model.compile("adam", "categorical_crossentropy")
model.fit([Ximg, Xseq], Y, epochs=5, batch_size=64)

def caption_image(path):
    feat = image_feature(path)
    seq = [tokenizer.word_index["startseq"]]
    for _ in range(max_len):
        x = pad_sequences([seq], maxlen=max_len)
        word_id = int(np.argmax(model.predict([feat[None, :], x], verbose=0)))
        word = tokenizer.index_word.get(word_id, "")
        if not word or word == "endseq": break
        seq.append(word_id)
    return " ".join(tokenizer.index_word.get(i, "") for i in seq[1:])

print(caption_image(os.path.join(IMAGE_DIR, next(iter(captions)))))
