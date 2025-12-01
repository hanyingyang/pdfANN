clear all
clc

%% Data generation script
% Generating inputs for Machine Learning
timestep = {'489', '801'};
filSize = size(timestep);
filSize = filSize(2);

FilterSize = fix(0.44 / 0.05);  % laminar premixed flame thermal thickness is 0.44 mm, and the uniform grid size is 0.05 mm
Delta = FilterSize+1;  % Odd filter size required by imboxfilt3
Jump = Delta*2;  % Moving distance along 1D of the filter, deciding number of data collected

Zst = 0.03; % stochiometric mixture fraction for hydrogen is 0.03

for ii=1:filSize
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
    
    Rho_bar = imboxfilt3(RHO,Delta);
    Z_tld = imboxfilt3(RHO.*Z,Delta) ./ Rho_bar;
    C_tld = imboxfilt3(RHO.*C,Delta) ./ Rho_bar;

    % Calculating second moments by q_var = q^2_tld - q_tld^2
    Z2_tld = imboxfilt3(RHO.*Z.^2,Delta) ./ Rho_bar;
    Zvar_tld = Z2_tld - Z_tld.^2;

    C2_tld = imboxfilt3(RHO.*C.^2,Delta) ./ Rho_bar;
    Cvar_tld = C2_tld - C_tld.^2;
    
    CZ_tld = imboxfilt3(RHO.*C.*Z,Delta) ./ Rho_bar;
    cov_tld = CZ_tld - C_tld.*Z_tld;
    
    W_bar = imboxfilt3(RHO.*W,Delta) ./ Rho_bar;
    
    cTld = C_tld((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    cVarTld = Cvar_tld((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    ZTld = Z_tld((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    ZVarTld = Zvar_tld((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    cZCovTld = cov_tld((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    rhoBar = Rho_bar((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    wBar = W_bar((Delta+1)/2:Jump:N_x-(Delta-1)/2,(Delta+1)/2:Jump:N_y-(Delta-1)/2,(Delta+1)/2:Jump:N_z-(Delta-1)/2);
    
    
    gZ = ZVarTld ./ (ZTld .* (1 - ZTld));
    gZ(ZTld==0|ZTld==1) = 0;
    ZVarTld(ZTld==0|ZTld==1) = 0;
    cZCovTld(ZTld==0|ZTld==1) = 0;
    %gZ(gZ<1e-6) = 0;
    %ZVarTld(gZ<1e-6) = 0;
    %cZCovTld(gZ<1e-6) = 0;
    
    gC = cVarTld ./ (cTld .* (1 - cTld));
    gC(cTld==0|cTld==1) = 0;
    cVarTld(cTld==0|cTld==1) = 0;
    cZCovTld(cTld==0|cTld==1) = 0;
    %gC(gC<1e-6) = 0;
    %cVarTld(gC<1e-6) = 0;
    
    time = sscanf(timestep{ii},'%s');
    filter_save=num2str(Delta);
    filter_jump=num2str(Jump);
    fprintf(strcat(['Saving data at time step ' time '\n']));
    fileDir = strcat(['./' timestep{ii} '/']);   
    save([fileDir strcat(['ts' timestep{ii} '_FS' filter_save '_Jump' filter_jump])],...
    'C','Z','RHO','W','cTld','ZTld','cVarTld','ZVarTld','cZCovTld','gC','gZ','rhoBar','wBar');

end