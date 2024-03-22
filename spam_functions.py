def processamento(file):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    
    spam = pd.read_csv(file)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(spam['Category'])
    mensagens = spam['Message'].values
    
    X_train, x_test, y_train, y_test = train_test_split(mensagens, y, test_size=0.3)
    return X_train, x_test, y_train, y_test

def tokenizer(X_train, x_test):
    from keras.preprocessing.text import Tokenizer 
    from keras.utils import pad_sequences
    
    token = Tokenizer(num_words=1000) #Cria objeto que vai extrair as 1000 palavras mais frequentes
    token.fit_on_texts(X_train) #Cria vocabulário para treino
    token.fit_on_texts(x_test) #Cria vocabulário para teste
    X_train = token.texts_to_sequences(X_train) #Converte mensagens para indices, no treino
    X_test = token.texts_to_sequences(x_test) #Converte mensagens para indices, no teste
    
    X_train = pad_sequences(X_train, padding='post', maxlen=500) 
    X_test = pad_sequences(X_test, padding='post', maxlen=500)
    
    
    return X_train, X_test, token


def rede_neural_treino(X_train, y_train, token):
    import pandas as pd
    from keras.models import Sequential 
    from keras.layers import Dense, Dropout, Flatten, Embedding
    
    
    
    #Criando o modelo de redes neurais
    modelo = Sequential() #Cria modelo de rede neural sequencial
    #Configurando o modelo de redes neurais
    modelo.add(Embedding(input_dim=len(token.word_index), output_dim=50, input_length=500)) 
    #Cria uma camada que madeia cada token para um vetor de 50 dimensões
    modelo.add(Flatten()) #Achata a saída para um vetor de 1 dimensão
    modelo.add(Dense(units=50, activation='sigmoid')) 
    #Adiciona camada de 50 neurônios, com Sigmoid de função ativação
    modelo.add(Dropout(0.1)) #Adiciona camada dropout para perda de 10%, evitando overfitting
    modelo.add(Dense(units=1, activation='sigmoid')) 
    #Adiciona 1 neurônio de saída com função de ativação sigmoid: Necessário para saídas binárias
    modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    #Compila o modelo, definindo alguns parâmetros. Erro, Otimizador, e a métrica
    modelo.summary()
    modelo.fit(X_train, y_train, epochs=50, batch_size=10, verbose=True, validation_data=(X_train, y_train))
    #Treina o modelo com base nos conjuntos de treino
    loss_history_treino = modelo.history.history['loss']
    accuracy_history_treino = modelo.history.history['accuracy']
    
    nova_previsao = modelo.predict(X_train)
    print(nova_previsao)
    prev_treino = (nova_previsao > 0.5)
    prev_treino = pd.DataFrame(prev_treino)
    print('Previsão de SPAM:')
    print(prev_treino)
    return prev_treino[:1672], modelo, loss_history_treino, accuracy_history_treino
    
def metrics_treino(y_train, prev_treino):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Calculando as métricas
    cm = confusion_matrix(y_train, prev_treino) 
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

    accuracy_treino = accuracy_score(y_train, prev_treino)
    accuracy_treino = np.round(accuracy_treino, 2)
    print("Acurácia:", accuracy_treino) #Acurácia

    precision = precision_score(y_train, prev_treino)
    precision = np.round(precision, 2)
    print("Precisão:", precision) #Precisão

    recall = recall_score(y_train, prev_treino)
    recall = np.round(recall, 2)
    print("Recall:", recall) #Sensibilidade

    f1 = f1_score(y_train, prev_treino)
    f1 = np.round(f1, 2)
    print("F1-Score:", f1) #F1-Score


    target_names = ['ham', 'spam']
    report  = classification_report(y_train, prev_treino, target_names=target_names)
    print(report) #Resumo métricas
    
    return accuracy_treino

def rede_neural_test(X_test, modelo):
    import pandas as pd
    
    
    nova_previsao = modelo.predict(X_test)
    print(nova_previsao)
    prev_test = (nova_previsao > 0.5) 
    prev_test = pd.DataFrame(prev_test)
    print('Previsão de SPAM:')
    print(prev_test)
    return prev_test


def metrics_test(y_test, prev_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Calculando as métricas
    cm = confusion_matrix(y_test, prev_test) 
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

    accuracy_test = accuracy_score(y_test, prev_test)
    accuracy_test = np.round(accuracy_test, 2)
    print("Acurácia:", accuracy_test) #Acurácia

    precision = precision_score(y_test, prev_test)
    precision = np.round(precision, 2)
    print("Precisão:", precision) #Precisão

    recall = recall_score(y_test, prev_test)
    recall = np.round(recall, 2)
    print("Recall:", recall) #Sensibilidade

    f1 = f1_score(y_test, prev_test)
    f1 = np.round(f1, 2)
    print("F1-Score:", f1) #F1-Score


    target_names = ['ham', 'spam']
    report  = classification_report(y_test, prev_test, target_names=target_names)
    print(report) #Resumo métricas
    
    return accuracy_test


def graficos(loss_history, accuracy_history_treino):
    import matplotlib.pyplot as plt
    
    plt.plot(loss_history)
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução de Loss')
    plt.show()
    
    
    plt.plot(accuracy_history_treino)
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.title('Evoluação da Acurácia')
    plt.show()