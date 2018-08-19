
#This is a program for detecting an increase in mean with an abs(N(gamma,tau^2))
#prior (gamma>0) by a  %SR procedure with known initial mean and constant known standard deviation=sigma. 
#Input:   x=row vector of observations, mean=mu, sd=sigma, gamma, tau, A
prog1 = function(x, mu, sigma, gamma, tau, A){

    r=0; 
    z=matrix((x-mu)/sigma,nrow=1); 
    len = length(z)

    zeta = pnorm(gamma/tau)
    w = matrix((10^(-10)) * cumsum(rep(1,len)), nrow = 1)
    t1 = matrix(c(1:len),nrow=1); 
    t2 = matrix(rep(1,len), nrow = 1);
    N1 = t(t1)%*%t2;
    N2 = t(N1);
    N3 = N1-N2+1+1/tau^2;
    N3[!lower.tri(N3,diag=TRUE)] = 0;

    X1 = t(matrix(cumsum(z),nrow=1))%*%t2
    X2 = t(X1)
    
    if(len>1){
        X3 = cbind( matrix(rep(0,len),ncol=1),  X2[1:len,1:(len-1)] );
    }
    else{
        X3 = 0;
    }

    X4 = X1 - X3 + gamma/tau^2;
    X4[!lower.tri(X4, diag = TRUE)] = 0;

    Y1 = sqrt(N3);
    
    Y2 = X4 / Y1;
    Y2[!lower.tri(Y2, diag = TRUE)] = 0;
    
    lambda = pnorm(Y2) * exp(0.5*(Y2^2) - 0.5*(gamma/tau)^2)/(Y1*tau*zeta);
    lambda[!lower.tri(lambda, diag = TRUE)] = 0;
    
    r=colSums(t(lambda));
    
    c=cumsum(pmax(r, A) - A) - w;
    
    d = min(c); 
    I=which.min(c);
    
    N=I+1;
    NUMmtLENGTH=max(N-len, 0);

    return(list(N = N, NUMmtLENGTH = NUMmtLENGTH))
}


x = read.csv('//hustaff.huji.local/dfs/sHome/ms/boazrm/work/random_normal.csv', header = FALSE, sep = ',');
colnames(x) = NULL;

x = as.numeric(unlist(x));
mu = 0;
sigma = 1;
gamma = 0;
tau = 0.5;
A = 100;
N = prog1(x, mu, sigma, gamma, tau, A)

N


