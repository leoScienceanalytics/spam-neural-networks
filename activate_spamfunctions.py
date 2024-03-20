from spam_functions import processamento, tokenizer, rede_neural, metrics

file = 'spam (1).csv'

X_train, x_test, y_train, y_test = processamento(file)
X_train, X_test, token = tokenizer(X_train, x_test)
prev = rede_neural(X_train, X_test, y_train, token)
metricas = metrics(y_test, prev)
print(metricas)

