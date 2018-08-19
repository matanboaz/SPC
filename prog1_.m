function [ N, NUMmtLENGTH ] = prog1( x, mu, sigma, gamma, tau, A )
    r = 0;
    z = (x - mu) / sigma;
    [o, length] = size(z);
    zeta = normcdf(gamma / tau);
    w = (10^(-10)) * cumsum( ones(1, length) );
    t1 = [1: length];  
    t2 = ones(1, length);
    N1 = t1' * t2;
    N2 = N1';
    N3 = tril( N1 - N2 + 1 + 1/tau^2 );
    
    X1 = (cumsum(z))' * t2;
    X2 = X1';
    if length > 1
        X3 = [ [zeros(length,1)] [X2(1: length,1:( length-1))] ];
    else
        X3 = 0;
    end
    
    X4 = tril(X1 - X3 + gamma/tau^2);
    Y1 = sqrt(N3);
    Y2 = tril(X4 ./ Y1);
    lambda = tril(normcdf(Y2).*exp(.5*Y2.^2-.5*(gamma/tau)^2)./(Y1*tau*zeta));
    r = sum(lambda');
    c = cumsum( max(r, A) - A ) - w;
    [d, I] = min(c);
    N = I + 1;
    NUMmtLENGTH = max(N - length, 0);
end

