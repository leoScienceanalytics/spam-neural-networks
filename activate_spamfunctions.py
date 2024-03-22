from spam_functions import processamento, tokenizer
from spam_functions import rede_neural_treino, rede_neural_test, metrics_treino, metrics_test
from spam_functions import graficos
file = 'spam (1).csv'

X_train, x_test, y_train, y_test = processamento(file)
X_train, X_test, token = tokenizer(X_train, x_test)

#Treinamento
prev_treino, modelo, loss_history_treino, accuracy_history_treino = rede_neural_treino(X_train, y_train, token)
metricas = metrics_treino(y_test, prev_treino)
print(metricas)

#Teste
prev_test = rede_neural_test(X_test, modelo)
print(prev_test)

perda = graficos(loss_history_treino, accuracy_history_treino)
print(perda)

