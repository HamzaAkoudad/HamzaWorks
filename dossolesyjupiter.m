clc;
clear;

% Constantes
G = 5168.6;   % Unidades: s = miles de km, t = horas, masa = masas terrestres
MJ = 318;     % Masa de Júpiter en masas terrestres
MS = 1e6;     % Masa del Sol en masas terrestres
Msat = 1e6;   % Masa de la estrella en masas terrestres

% Velocidades y posiciones iniciales
vS0 = [0; 90];      rS0 = [-0.9e5; 0];
vJ0 = [0; 60];      rJ0 = [8e5; 0];
vsat0 = [0; -90];   rsat0 = [0.9e5; 0];

% Vector de estado inicial
w0 = [rJ0; rsat0; rS0; vJ0; vsat0; vS0];

% Tiempo de simulación
tmax = 24 * 30 * 12 * 12;  % 12 años en horas
num_pasos = 1e6;           % Número de pasos
t_eval = linspace(0, tmax, num_pasos); % Puntos de evaluación

% Resolver ecuaciones diferenciales
f = @(t, W) termino_dcha_tres(t, W, G, MS, MJ, Msat);
[tt, WW] = odeRK4(f, 0, tmax, w0, num_pasos);

% Extraer posiciones
rJ = WW(1:2, :);    % Posición de Júpiter
rsat = WW(3:4, :);  % Posición de la estrella
rS = WW(5:6, :);    % Posición del Sol

% Graficar trayectoria
figure;
hold on;
axis equal;
axis([-9e5, 9e5, -9e5, 9e5]);
grid on;

for i = 1:1000:num_pasos
    clf;
    hold on;
    
    % Dibujar posiciones actuales
    plot(rS(1, i), rS(2, i), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8); % Sol
    plot(rJ(1, i), rJ(2, i), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6); % Júpiter
    plot(rsat(1, i), rsat(2, i), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 4); % Satélite
    
    % Dibujar trayectorias
    plot(rS(1, 1:i), rS(2, 1:i), 'r', 'LineWidth', 1);
    plot(rJ(1, 1:i), rJ(2, 1:i), 'k', 'LineWidth', 1);
    plot(rsat(1, 1:i), rsat(2, 1:i), 'b', 'LineWidth', 1);
    
    % Ajustar ejes
    axis equal;
    axis([-9e5, 9e5, -9e5, 9e5]);
    grid on;
    
    pause(0.01);  % Pausa para visualizar la animación
end

hold off;
