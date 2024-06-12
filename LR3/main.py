import requests
from bs4 import BeautifulSoup

# URL страницы, с которой нужно загрузить текст
url = "https://hpmor.ru/files/hpmor_ru.html"

# Отправляем GET запрос на получение содержимого страницы
response = requests.get(url)

# Создаем объект BeautifulSoup для парсинга HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Ищем все теги <p> (параграфы) и извлекаем из них текст
paragraphs = soup.find_all('p')

# Объединяем текст из всех параграфов в одну строку
text = '\n'.join([p.text for p in paragraphs])

# Теперь переменная text содержит текст страницы
print(text[:100])



# Импортируем необходимые библиотеки
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import GPT2Tokenizer

# Токенизация текста для двунаправленной LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

# Преобразование текста в последовательности слов
maxlen = 10  # длина последовательности
step = 1  # шаг сдвига
sentences = []
next_words = []

for i in range(0, len(sequences) - maxlen, step):
    sentences.append(sequences[i: i + maxlen])
    next_words.append(sequences[i + maxlen])

# Преобразование в числовые массивы
X = np.array(sentences)
y = np.array(next_words)
y = keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Токенизация текста для GPT
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
inputs = gpt_tokenizer(text, return_tensors="tf", max_length=512, truncation=True)


from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding
from keras.utils import to_categorical

# Создание модели
model_bilstm = Sequential()
model_bilstm.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=maxlen))
model_bilstm.add(Bidirectional(LSTM(128)))
model_bilstm.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

model_bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model_bilstm.fit(X, y, batch_size=128, epochs=20)
