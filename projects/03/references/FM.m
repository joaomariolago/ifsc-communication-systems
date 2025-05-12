pkg load signal;
pkg load communications;


fc = 1e3; % frequencia da portadora
Tc = 1/fc; % perido da onda portadora
KF = 5*pi*100; %500*pi -> beta=5  pois fm=100
Ac = 1; %amplitude da portadora

Ts = Tc/50; %intervalo de amostragem
t = 0:Ts:100*Tc; % intervalo de tempo dos sinais

fm = 100; %frequencia do sinal de informacao (sinal modulante)
Am=2; %amplitude do sinal de informacao (sinal modulante)
m = Am*sin(2*pi*fm*t); % sinal de informação senoidal

beta= (KF*Am)/(2*pi*fm); %indice de modulacao
bw=2*(beta+1)*fm; % Largura de banda - Carson

%Integrando o sinal de informacao - Método do Trapézio
intm = zeros(size(t));
for k = 2:length(t)
    intm(k) = intm(k-1) + 0.5*Ts*(m(k-1) + m(k));
end

%Portadora
c = Ac*cos(2*pi*fc*t);

theta = 2*pi*fc*t + KF*intm; %angulo
fi = diff(theta)/(Ts*2*pi); %frequencia instantanea

s = Ac*cos(theta); %Sinal FM

figure;
subplot(2,2,1)
plot(t,s);

subplot(2,2,2)
plot(t,m/2,'r')
title(['Sinal modulado no dominio do tempo. Sensibilidade do modulador = ',num2str(KF/(2*pi)),'Hz/V']);
legend('Sinal FM','Informação');


subplot(2,2,3)
plot_fft(Ts,s,0,2000,'b');
title(['Espectro do sinal modulado em FM. Indice de modulação (beta) = ',num2str(beta),'. BW Carson = ',num2str(bw),'Hz']);
