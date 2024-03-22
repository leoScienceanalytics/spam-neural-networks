from spam_functions import processamento, tokenizer, rede_neural_treino, metrics_treino, rede_neural_test

file = 'spam (1).csv'

X_train, x_test, y_train, y_test = processamento(file)
X_train, X_test, token = tokenizer(X_train, x_test)

#Treinamento
prev_treino, modelo = rede_neural_treino(X_train, y_train, token)
metricas = metrics_treino(y_test, prev_treino)
print(metricas)

#Teste
prev_test = rede_neural_test(X_test, modelo)
print(prev_test)


