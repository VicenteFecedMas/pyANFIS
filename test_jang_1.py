import torch

from main import ANFIS

x_dim = 100
y_dim = 5
z_dim = 2

x = torch.empty(x_dim, y_dim, z_dim)
y = torch.empty(x_dim, y_dim, 1)

for b in range(x_dim):
    x[b, :, :] = torch.linspace(1, y_dim*z_dim, y_dim*z_dim).view(y_dim, z_dim)
    #print(x[b, :, :])
    y[b, :, 0] = (torch.sin(x[b, :, 0]) * torch.sin(x[b, :, 1])) / (x[b, :, 0]*x[b, :, 1])
    #print(y[b, :, 0])
    #break



n_epochs = 100
b_size = 10
b, j, i = x.size()

anfis = ANFIS(x, y)
anfis.consequents.output_dim = (b_size, y_dim, 1)


print("Training start")
anfis.train()
for epoch in range(n_epochs):
    if epoch == 0 or epoch % 5 == 0:
        anfis.antecedents.display(list(anfis.antecedents.universes))

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
                        lr = 10e-5

                        if param.grad is not None:
                            # Calculate the L2 norm of the gradients
                            grad_norm = torch.norm(param.grad)

                            # Scale the gradients if the norm is too small
                            scale = max(1.0, grad_norm / 0.01)  # You can adjust the threshold (0.01) as needed

                            # Apply gradient scaling
                            scaled_grad = param.grad / scale

                            # Update the parameters
                            with torch.no_grad():
                                param.data -= scaled_grad * lr
        
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