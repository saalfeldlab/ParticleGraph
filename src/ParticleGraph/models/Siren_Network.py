# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren_Network(nn.Module):
    def __init__(self, image_width, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., device='cpu'):
        super().__init__()

        self.device = device 
        self.image_width = image_width
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.net = self.net.to(device)

    @property
    def values(self):
        # Call forward method
        output, coords = self.__call__()
        return output.squeeze().reshape(self.image_width, self.image_width)
    
    def coordinate_grid(self, n_points):
        coords = np.linspace(0, 1, n_points, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        return xy_grid
    
    def get_mgrid(self, sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def forward(self, coords=None, time=None):
        if coords is None:
            coords = self.get_mgrid(self.image_width, dim=2).to(self.device)
            if time != None:
               coords = torch.cat((coords, time * torch.ones_like(coords[:, 0:1])), 1)

        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output


            # print('Train SIREN model ...')
            #
            # run = 1 # + np.random.randint(NGraphs - 1)
            # frame = 62 #+ np.random.randint(n_frames - 2)
            #
            # image_width = 256
            # model_input = get_mgrid(image_width, 2)
            # model_input = model_input.cuda()
            #
            # for frame in range(62,63): # range(0,n_frames,50):
            #
            #
            #     model_siren = Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=256, hidden_layers=8, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.)
            #     x = x_list[run][frame].clone().detach()
            #
            #     pos_coords = x[:, 1:3].clone().detach() * image_width
            #     t_coords = k / n_frames * torch.ones_like(pos_coords[:, 0:1], device=device)
            #     coords = pos_coords  # torch.cat((pos_coords, t_coords), 1)
            #     y = x[:, 3:5].clone().detach() / vnorm
            #
            #     # image_file = './graphs_data/pattern_1.tif'  # boat_512.tif, beads_abb.tif, beads_gt.tif
            #     # target = load_image(image_file, crop_width=image_width, device=device)
            #
            #     target = torch.zeros((2, image_width, image_width), device=device)
            #     target[0,pos_coords[:, 0].long(), pos_coords[:, 1].long()] = -y[:,0].squeeze()
            #     target[0:1,:,:] = GaussianBlur(51,10)(target[0:1,:,:])
            #     target[1,pos_coords[:, 0].long(), pos_coords[:, 1].long()] = y[:,1].squeeze()
            #     target[1:2,:,:] = GaussianBlur(51,10)(target[1:2,:,:])
            #
            #     # matplotlib.use("Qt5Agg")
            #     # fig = plt.figure(figsize=(8, 8))
            #     # plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=20, c=to_numpy(y[:,0].squeeze()))
            #     # fig = plt.figure(figsize=(8, 8))
            #     # plt.imshow(to_numpy(target[0, :, :].squeeze()))
            #     # plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=20, c=to_numpy(y[:, 0].squeeze()))
            #
            #     total_steps = 1000
            #     steps_til_summary = 200
            #     optim = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())
            #
            #     for step in trange(total_steps + 1):
            #         model_output, coords = model_siren()
            #         img_grad_ = gradient(model_output, coords)
            #         # loss = ((model_output - ground_truth) ** 2).mean()
            #         loss = (img_grad_[:,1].view(image_width, image_width) - target[0]).norm(2) + (img_grad_[:,0].view(image_width, image_width) - target[1]).norm(2)
            #
            #         # if not step % steps_til_summary:
            #         #     print("Step %d, Total loss %0.6f" % (step, loss))
            #
            #         optim.zero_grad()
            #         loss.backward()
            #         optim.step()
            #
            #     # matplotlib.use("Qt5Agg")
            #
            #     fig = plt.figure(figsize=(16, 8))
            #     ax = fig.add_subplot(2, 4, 2)
            #     plt.imshow(to_numpy(target[0, :, :].squeeze()))
            #     plt.title('Velocity_field_y')
            #     plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
            #     plt.xlim([0, image_width])
            #     plt.ylim([0, image_width])
            #     plt.xticks([])
            #     plt.yticks([])
            #     ax.invert_yaxis()
            #     ax = fig.add_subplot(2, 4, 1)
            #     plt.imshow(to_numpy(target[1, :, :].squeeze()))
            #     plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
            #     plt.xlim([0, image_width])
            #     plt.ylim([0, image_width])
            #     plt.xticks([])
            #     plt.yticks([])
            #     ax.invert_yaxis()
            #     plt.title('Velocity_field_x')
            #     ax = fig.add_subplot(1, 2, 2)
            #     temp = -model_output.cpu().view(image_width,image_width).permute(1,0).detach().numpy()
            #     plt.imshow(temp)
            #     plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=1, color='w')
            #     plt.xlim([0, image_width])
            #     plt.ylim([0, image_width])
            #     plt.xticks([])
            #     plt.yticks([])
            #     ax.invert_yaxis()
            #     plt.title('Reconstructed potential')
            #     ax = fig.add_subplot(2, 4, 6)
            #     plt.imshow(img_grad_[:, 1].cpu().view(image_width, image_width).detach().numpy())
            #     plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
            #     plt.xlim([0, image_width])
            #     plt.ylim([0, image_width])
            #     plt.xticks([])
            #     plt.yticks([])
            #     ax.invert_yaxis()
            #     plt.title('Gradient_y from potential')
            #     ax = fig.add_subplot(2, 4, 5)
            #     plt.imshow(img_grad_[:, 0].cpu().view(image_width, image_width).detach().numpy())
            #     plt.scatter(to_numpy(pos_coords[:, 1]), to_numpy(pos_coords[:, 0]), s=0.1, color='w')
            #     plt.xlim([0, image_width])
            #     plt.ylim([0, image_width])
            #     plt.xticks([])
            #     plt.yticks([])
            #     ax.invert_yaxis()
            #     plt.title('Gradient_x from potential')
            #     plt.tight_layout()
            #     plt.savefig(f"{log_dir}/tmp_training/siren/siren_{frame}.jpg", dpi=170.7)
            #     plt.close()
            #
            # torch.save({'model_state_dict': model_siren.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict()},
            #            # os.path.join(log_dir, 'models', f'Siren_model'))
