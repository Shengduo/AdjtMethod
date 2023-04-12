clc,clear;
close all;
load('./plots/FricSeqGen_Plot2Params_ab/res.mat');
xlab = "$a$";
ylab = "$b$";
nOfGridPts = 50;
xRange = [0.001, 0.021];
yRange = [0.001, 0.021];
% param1s = log10(param1s);

fig1 = figure(1);
fig1.Position(3:4) = 2 * fig1.Position(3:4);

p = pcolor(param1s, param2s, AllOs);
p.EdgeColor = "None";
p.FaceColor = 'interp';

% set(gca, 'XScale', 'log');
xlabel(xlab, "FontSize", 20, "Interpreter", "latex");
ylabel(ylab, "FontSize", 20, "Interpreter", "latex");
[ind, minO] = min(AllOs);
set(gca, "FontSize", 20);
clim([0, 0.01]);
c = colorbar;
set(gcf, 'color', 'w');
ylabel(c, "Objective Value", "FontSize", 20, 'Interpreter','latex');

hold on;

% Plot the vector field
hx = (xRange(2) - xRange(1)) / (nOfGridPts - 1);
hy = (yRange(2) - yRange(1)) / (nOfGridPts - 1);
[grady, gradx] = gradient(AllOs, hy, hx);
grad_inds = 1 : floor(nOfGridPts / 20) : nOfGridPts;
quiver(param1s(grad_inds, grad_inds), ...
       param2s(grad_inds, grad_inds), ...
       -gradx(grad_inds, grad_inds), ...
       -grady(grad_inds, grad_inds), ...
       'color', 'r');

% %% Initial, final beta and target beta
% target_params = [log10(1. / 1.e1), 0.58];
% 
% % initial_params = [log10(1. / 5.e1), 0.7]; % for not log gradient, data1
% % final_params = [log10(0.4823), 0.5758];  % for not log gradient, data1
% initial_params = [log10(1.), 0.2]; % For log and not log gradient, data2
% % final_params = [log10(0.9939), 0.5751];  % For not log gradient, data2
% final_params = [0.0121, 0.5750]; % For log gradient, data2
% h = zeros(1, 3);
% h(1) = scatter(initial_params(1), initial_params(2), 'k', 'filled', 'SizeData', 50);
% h(3) = scatter(target_params(1), target_params(2), 'r', 'filled', 'SizeData', 50);
% h(2) = scatter(final_params(1), final_params(2), 'y', 'filled', 'SizeData', 50);
% 
% legend(h, ["Initial", "Final", "Target"]);