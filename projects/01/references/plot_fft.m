function [freq, amp] = plot_fft (Ts,sinal,fmin,fmax,plot_arg)
    sinal = [sinal zeros(1,1e6)];

    ftt = fftshift(abs(fft(sinal))/numel(sinal));

    f = (-numel(ftt)/2):(numel(ftt)/2-1);
    f = f.*(1/Ts)/numel(ftt);

    plot(f,ftt,plot_arg);

    axis([fmin fmax 0 max(ftt)*1.1]);

end
