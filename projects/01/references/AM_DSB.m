Tw = .001;
Ts = 1e-8;

Fc = 160e3;
Ac = 5;

Fm = 4e3;
Am = 2;

t = 0:Ts:Tw;

ct = Ac*cos(2*pi*t*Fc);
mt = Am*cos(2*pi*t*Fm);
k=1/Ac;

st = (1 + k.*mt).*ct;

figure;
subplot(2,2,1);
plot(t,ct);
title('Portadora');

subplot(2,2,2);
plot(t,mt);
title('Modulante');

subplot(2,2,3);
plot(t,st);
title(['Modulado (constante de sensibilidade = ', num2str(k), ', indice de modulação = ', num2str(Am/Ac),')']);

subplot(2,2,4);
plot_fft(Ts,st,130e3,190e3,'r');
title('Sinais no dominio da frequencia');



