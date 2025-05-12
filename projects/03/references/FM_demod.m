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

sFM = Ac*cos(theta); %Sinal FM


% Processo de Demodulação FM

hilbert_t = hilbert(sFM); % Transformada de Hilbert para obter a fase instantânea
instantaneous_phase = unwrap(angle(hilbert_t)); % Desfase do sinal para recuperar a fase
demodulated_signal = diff(instantaneous_phase) / Ts; % Derivada para obter o sinal demodulado
demodulated_signal = [demodulated_signal, demodulated_signal(end)]; % Ajuste de tamanho



% Plots

figure;
subplot(3, 1, 1);
plot(t, m);
title('Sinal Original (Modulante) no Domínio do Tempo');
xlabel('Tempo (s)');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(t, sFM);
title('Sinal FM Modulado no Domínio do Tempo');
xlabel('Tempo (s)');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(t, demodulated_signal);
title('Sinal Demodulado no Domínio do Tempo');
xlabel('Tempo (s)');
ylabel('Amplitude');

% Análise no domínio da frequência

figure;
subplot(3, 1, 1);
M_f = abs(fftshift(fft(m)));
freq = linspace(-fc, fc, length(M_f));
plot(freq, M_f);
title('Espectro do Sinal Original');
xlabel('Frequência (Hz)');
ylabel('Magnitude');


subplot(3, 1, 2);
SFM_f = abs(fftshift(fft(sFM)));
plot(freq, SFM_f);
title('Espectro do Sinal Modulado FM');
xlabel('Frequência (Hz)');
ylabel('Magnitude');


subplot(3, 1, 3);
Demod_f = abs(fftshift(fft(demodulated_signal)));
plot(freq, Demod_f);
title('Espectro do Sinal Demodulado');
xlabel('Frequência (Hz)');
ylabel('Magnitude');
