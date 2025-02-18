clc;
clear;

% Constants
G = 5168.6;   % Gravitational constant in custom units (s = thousands of km, t = hours, mass = Earth masses)
MJ = 318;     % Jupiter's mass in Earth masses
MS = 1e6;     % Sun's mass in Earth masses
Msat = 1e6;   % Star's mass in Earth masses

% Initial velocities and positions
vS0 = [0; 90];      rS0 = [-0.9e5; 0];  % Sun
vJ0 = [0; 60];      rJ0 = [8e5; 0];     % Jupiter
vsat0 = [0; -90];   rsat0 = [0.9e5; 0]; % Satellite

% Initial state vector
w0 = [rJ0; rsat0; rS0; vJ0; vsat0; vS0];

% Simulation time
tmax = 24 * 30 * 12 * 12;  % 12 years in hours
num_pasos = 1e6;           % Number of time steps
t_eval = linspace(0, tmax, num_pasos); % Time evaluation points

% Solve differential equations
f = @(t, W) termino_dcha_tres(t, W, G, MS, MJ, Msat);
[tt, WW] = odeRK4(f, 0, tmax, w0, num_pasos);

% Extract positions
rJ = WW(1:2, :);    % Jupiter's position
rsat = WW(3:4, :);  % Satellite's position
rS = WW(5:6, :);    % Sun's position

% Plot trajectory
figure;
hold on;
axis equal;
axis([-9e5, 9e5, -9e5, 9e5]);
grid on;

for i = 1:1000:num_pasos
    clf; % Clear figure
    hold on;
    
    % Plot current positions
    plot(rS(1, i), rS(2, i), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8); % Sun
    plot(rJ(1, i), rJ(2, i), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6); % Jupiter
    plot(rsat(1, i), rsat(2, i), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 4); % Satellite
    
    % Plot trajectories
    plot(rS(1, 1:i), rS(2, 1:i), 'r', 'LineWidth', 1);
    plot(rJ(1, 1:i), rJ(2, 1:i), 'k', 'LineWidth', 1);
    plot(rsat(1, 1:i), rsat(2, 1:i), 'b', 'LineWidth', 1);
    
    % Adjust axes
    axis equal;
    axis([-9e5, 9e5, -9e5, 9e5]);
    grid on;
    
    pause(0.01);  % Pause for animation effect
end

hold off;
