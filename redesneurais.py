import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, Embedding 
from keras.preprocessing.text import Tokenizer 
from keras.utils import pad_sequences



# Exemplo de previsões e valores reais

spam = pd.read_csv('spam (1).csv')

labelencoder = LabelEncoder() 
#Cria um objeto para converter rótulos categóricos na coluna "Categoria" em valores numéricos
y = labelencoder.fit_transform(spam['Category']) 
#Aplica codificador a coluna categoria
print(y)

mensagens = spam['Message'].values #Extrai valores da conluna Message

X_train, x_test, y_train, y_test = train_test_split(mensagens, y, test_size=0.3)

token = Tokenizer(num_words=1000) #Cria objeto que vai extrair as 1000 palavras mais frequentes
token.fit_on_texts(X_train) #Cria vocabulário para treino
token.fit_on_texts(x_test) #Cria vocabulário para teste
X_train = token.texts_to_sequences(X_train) #Converte mensagens para indices, no treino
X_test = token.texts_to_sequences(x_test) #Converte mensagens para indices, no teste

print(X_train)

X_train = pad_sequences(X_train, padding='post', maxlen=500) 
X_test = pad_sequences(X_test, padding='post', maxlen=500)
#Preenche sequências para torná-las  mais curtas

#Criando o modelo de redes neurais
modelo = Sequential() #Cria modelo de rede neural sequencial
#Configurando o modelo de redes neurais
#Camada de entrada
modelo.add(Embedding(input_dim=len(token.word_index), output_dim=50, input_length=500)) 
#Cria uma camada que madeia cada token para um vetor de 50 dimensões
#Camada Oculta
modelo.add(Flatten()) #Achata a saída para um vetor de 1 dimensão
modelo.add(Dense(units=10, activation='relu')) 
#Adiciona camada de 10 neurônios, com ReLu de função ativação
modelo.add(Dropout(0.1)) #Adiciona camada dropout para perda de 10%, evitando overfitting
#Camada de Saída
modelo.add(Dense(units=1, activation='sigmoid')) 
#Adiciona 1 neurônio de saída com função de ativação sigmoid: Necessário para saídas binárias

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
#Compila o modelo, definindo alguns parâmetros. Erro, Otimizador, e a métrica
modelo.summary()
modelo.fit(X_train, y_train, epochs=20, batch_size=10, verbose=True, validation_data=(X_train, y_train))
#Treina o modelo com base nos conjuntos de treino

nova_previsao = modelo.predict(X_test)
print(nova_previsao)
prev = (nova_previsao > 0.5) 
prev = pd.DataFrame(prev)
print('Previsão de SPAM:')
print(prev)

# Calculando as métricas
cm = confusion_matrix(y_test, prev) 
print('Matriz de Confusão:')
print(cm)#Matriz de Confusão
#Plotando a figura
fig = plt.figure(figsize=(5,5))
ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
#Condigurações da figura
ax.set_title('Matriz de Confusão - SPAM\n\n')
ax.set_xlabel('\nValores Previstos')
ax.set_ylabel('Valores Reais')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['True','False'])
plt.show() #Matriz de Confusão

accuracy = accuracy_score(y_test, prev)
accuracy = np.round(accuracy, 2)
print("Acurácia:", accuracy) #Acurácia

precision = precision_score(y_test, prev)
precision = np.round(precision, 2)
print("Precisão:", precision) #Precisão

recall = recall_score(y_test, prev)
recall = np.round(recall, 2)
print("Recall:", recall) #Sensibilidade

f1 = f1_score(y_test, prev)
f1 = np.round(f1, 2)
print("F1-Score:", f1) #F1-Score


target_names = ['ham', 'spam']
report  = classification_report(y_test, prev, target_names=target_names)
print(report) #Resumo métricas