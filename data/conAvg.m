clear all
clc

%% Generating conditional averaging quantities across all snapshots
timestep = {'489'};
filSize = size(timestep);
filSize = filSize(2);

% Number and allocation of the discretisation points along c and Z, keeping
% same as in targetData.py
Zst=0.03;
nbins_C = 41;
nbins_Z = 61;
nZ1 = (nbins_Z - 1) * (2/3);
nZ2 = nbins_Z - 2 + 1 - nZ1;
edges = cell(2,1);
edges{1} = linspace(0,1,nbins_C);
Zspace1 = linspace(-5,0.3,nZ1);
Zspace2 = linspace(0.3,3,nZ2);
edges{2} = [-6,Zspace1,Zspace2(2:end),log(1/Zst)];
edges{2} = Zst * exp(edges{2});

% matrix of conditioning average and number of points
W_bins = zeros(nbins_C,nbins_Z);
npoints = zeros(nbins_C,nbins_Z);

for ii = 1:filSize
    fil = strcat([timestep{ii} '/ts' timestep{ii} '.h5']);  
    text=strcat(['Processing time step: ' timestep{ii} '\n']);
    fprintf(text);
    
    RHO = h5read(fil,'/RHO');
    W = h5read(fil,'/omega_c');
    C = h5read(fil,'/c');
    Z = h5read(fil,'/xi');
    
    [N_x, N_y, N_z] = size(Z); % x is the normal direction, y is the streamwise direction and z is the spanwise direction
   
    Z(Z<0) = 0; Z(Z>1) = 1;
    C(C<0) = 0; C(C>1) = 1;
    
    C_1D = reshape(C,1,[]);
    Z_1D = reshape(Z,1,[]);
    W_1D = reshape(W,1,[]);
    Rho_1D = reshape(RHO,1,[]);

    % Calculating conditional mean reaction rate
    for i = 1:length(C_1D(:))
        idx_C = find(edges{1}>C_1D(i),1,'first');
        if isempty(idx_C)
            idx_C = nbins_C + 1;
        end
 
        idx_Z = find(edges{2}>Z_1D(i),1,'first');
        if isempty(idx_Z)
            idx_Z = nbins_Z + 1;
        end
        if idx_Z == 1
            idx_Z = idx_Z + 1;
        end
    
        W_bins(idx_C-1,idx_Z-1) = W_bins(idx_C-1,idx_Z-1) + W_1D(i)/Rho_1D(i);
        npoints(idx_C-1,idx_Z-1) = npoints(idx_C-1,idx_Z-1) + 1;
    end
end
W_avg = W_bins ./ npoints;

fileDir = './';
save([fileDir strcat(['conAvg_WcY_H2_Lifted_' num2str(nbins_C) 'c_' num2str(nbins_Z) 'Z'])],'W_avg');