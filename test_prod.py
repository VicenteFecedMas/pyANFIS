import torch
from tabulate import tabulate

def print_results(x, y, output):

    num_cols = x.size(-1)
    names, data = [], []
    for i in range(num_cols):
        names.append(f"Tensor {i}")
        data.append(x[:, :, i].tolist()[0])

    num_cols = y.size(-1)
    for i in range(num_cols):
        names.append(f"Result {i}")
        data.append(y[:, :, i].tolist()[0])

    num_cols = output.size(-1)
    for i in range(num_cols):
        names.append(f"Output {i}")
        data.append(output[:, :, i].tolist()[0])

    print(tabulate(torch.tensor(data).T, headers=names, tablefmt="fancy_grid"))


import torch

from main import ANFIS
lr = 1

x_dim = 100
y_dim = 1000
z_dim = 2

x = torch.empty(x_dim, y_dim, z_dim)
y = torch.empty(x_dim, y_dim, 1)

for b in range(x_dim):
    x[b, :, :] = torch.randint(low=1, high=100, size=(y_dim, z_dim))
    #y[b, :, 0] = torch.sum(x[b, :, :], axis=1)
    y[b, :, 0] = torch.prod(x[b, :, :], axis=1)
    #y[b, :, 0] = torch.mean(x[b, :, :], axis=1)

n_epochs = 100
b_size = 2
b, j, i = x.size()

anfis = ANFIS(x, y)
anfis.consequents.output_dim = (b_size, y_dim, 1)


print("Training start")
anfis.train()
for epoch in range(n_epochs):
    #if epoch == 0 or epoch % 5 == 0:
    #    anfis.antecedents.display(list(anfis.antecedents.universes))

    total_loss = 0
    print(f'''  Epoch {epoch}''')
    for i in range(int(b/ b_size)):
        x_train = x[i*b_size:(i+1)*b_size, :, :]
        y_train = y[i*b_size:(i+1)*b_size, :, :]

        # Forward pass
        output = anfis(x_train, y_train)

        loss = torch.nn.MSELoss()(output, y_train)

        # Backward pass
        loss.backward()
        anfis.zero_grad()

        #print_results(x_train, y_train, output)
        #user_input = input() 

        # Update parameters
        for key, universe in anfis.antecedents.universes.items():
            for name, function in universe.universe.items():
                parameters = list(function.parameters())

                if parameters:
                    grads = torch.tensor([param.grad.data for param in parameters])
                    for param in parameters:
                        grad_norm = torch.sum(grads)
                        
                        with torch.no_grad():
                            param.data -= lr*param.grad.data/grad_norm
        
        total_loss += loss

        

    print(f'''       Epoch Loss {total_loss/(b/ b_size)}''')
    if total_loss < 0.0001:
            print(f'''      Early stopping''')
            break

anfis.eval()
x_prueba = torch.rand(1, y_dim, z_dim)

print(f'''
      PARA LOS NUMEROS :{x_prueba}
      OBTENEMOS: {anfis(x_prueba)}
      Y: {torch.prod(x_prueba, axis=2)}''')