import torch

from main import ANFIS

x_dim = 100
y_dim = 5
z_dim = 2

x = torch.empty(x_dim, y_dim, z_dim)
y = torch.empty(x_dim, y_dim, 1)

for b in range(x_dim):
    x[b, :, :] = torch.linspace(0, y_dim*z_dim, y_dim*z_dim).view(y_dim, z_dim)
    y[b, :, 0] = torch.sum(x[b, :, :], axis=1)
    #y[b, :, 0] = torch.prod(x[b, :, :], axis=1)
    #y[b, :, 0] = torch.mean(x[b, :, :], axis=1)

n_epochs = 100
b_size = 10
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

        #print(f'''   {i*b_size}-{(i+1)*b_size} Loss {loss}''')

        # Update parameters
        
        for key, universe in anfis.antecedents.universes.items():
            for name, function in universe.universe.items():
                parameters = list(function.parameters())

                if parameters:
                    for param in parameters:
                        print("PARAMETERS", param)
                        lr = 10e-5
                        with torch.no_grad():
                            param.data -= param.grad.data/lr
        
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
      Y: {torch.sum(x_prueba, axis=2)}''')