clc
clear
G=5168.6;%s=miles de km, t=horas, masa=masa tierras
MJ=318;
MS=1e6;
Msat=1e6;
%Velocidades y posiciones(W)
f=@(t,W) termino_dcha_tres(t,W,G,MS,MJ,Msat)
vS0=[0;90];rS0=[-0.9e5;0];
vJ0=[0;60];rJ0=[8e5;0];
vsat0=[0;-90];rsat0=[0.9e5;0];
w0=[rJ0;rsat0;rS0;vJ0;vsat0;vS0];
tmax=24*30*12*12;
[tt,WW]=odeRK4(f,0,tmax,w0,1000000);
 rJ=WW(1:2,:);
 rsat=WW(3:4,:);
 rS=WW(5:6,:);
for i=1:1000:1000000
plot(rS(1,i),rS(2,i),'ro',rJ(1,i),rJ(2,i),'ko',rsat(1,i),rsat(2,i),'bo')
hold on
plot(rS(1,1:i),rS(2,1:i),'r',rJ(1,1:i),rJ(2,1:i),'k',rsat(1,1:i),rsat(2,1:i),'b')
axis([-9e5,9e5,-9e5,9e5])
shg
hold off
end