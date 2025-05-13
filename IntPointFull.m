function [x, lambda, mu, z, norma, cinter] = IntPointFull(Q,A,F,d,b,c)
% Método de puntos interiores para el problema cuadrático
% Min   (0.5)* x' * Q * x + d´* x
% s.a.   A * x = b
%        F*x >= c
% 
% El sistema lineal de Newton se resuelve completo.
%
% In
% Q.- matriz nxn simétrica y positiva definida
% A.- matriz mxn con m <= n y rango(A) = m.
% F.- matriz pxn.
% d.- vector columna en R^n .
% b.- vector columna en R^m .
% c.- vector columna en R^p .
%
%Out
% x.- vector en R^n con la aproximación del mínimo local.
% lambda.- vector en R^m con la aproximación al multiplicador de Lagrange 
%          asociado a la restricción de igualdad.
% mu.- vector en R^p con la aproximación al multiplicador de Lagrange 
%      asociado a la restricción de desigualdad.
% z.- vector en R^p con la variable de holgura en la restricción de 
%     desigualdad.
% norma .- el valor de la norma euclidiana de las condiciones de KKT en la
%          en la última iteración
% cinter .- número de iteraciones del método de puntos interiores

% Parámetros iniciales
tol = 1e-06;
maxiter = 100;
cinter = 0;
%-----------------------------------------------------------
n = length(d);
m = length(b);
p = length(c);
%-----------------------------------------------------------
x = ones(n,1);
lambda = zeros(m,1);
mu = ones(p,1);
z = F*x - c + (0.5)*ones(p,1);
tau = (0.5)*(mu'*z)/p;
%-----------------------------------------------------------
% Norma de las condiciones necesarias de KKT
H =[Q*x+A'*lambda-F'*mu+d; A*x-b;-F*x+z+c;mu.*z];
norma = norm(H, "inf");
disp('Iter      CNPO           alfa    tau    rcond(K)')
disp('-----------------------------------------')
while(norma > tol && cinter < maxiter)
    cinter = cinter+1;

    % Resuelve el sistema lineal de Newton para la trayectoria
    % central
    K = [ Q            A'          -F'     zeros(n,p);
          A         zeros(m)    zeros(m,p) zeros(m,p);
         -F         zeros(p,m)  zeros(p)   eye(p);
       zeros(p,n)   zeros(p,m)  diag(z)    diag(mu)];
      
    condK = rcond(K);
    ld =-[Q*x+A'*lambda-F'*mu+d; A*x-b;-F*x+z+c;mu.*z-tau];
    
    % Sistema Lineal a resolver
    y = K\ld;
    %-----------------------------------------------------------
    % Se calculan los pasos
    Dx = y(1:n);
    Dlambda = y(n+1:n+m);
    
    Dmu = y(n+m+1:n+m+p);
    Dz = y(n+m+p+1:n+m+p+p);
    %-----------------------------------------------------------
    % Acorta el paso
    bt = ones(p,1); gm = ones(p,1);
    for k = 1:p
        if(Dmu(k) < 0)
            bt(k) = -(mu(k)/Dmu(k));
        end
        if(Dz(k) < 0)
            gm(k) =-(z(k)/Dz(k));
        end 
    end
    
    alfa =(0.995)*min([1 ; bt; gm]);  
    %-----------------------------------------------------------
    % Nuevo punto
    x = x + alfa*Dx;
    lambda = lambda + alfa*Dlambda;
    mu = mu + alfa*Dmu;
    z  = z + alfa*Dz;
    %-----------------------------------------------------------
    % Nueva tau
    tau = (0.5)*(mu'*z)/p;
    %----------------------------------------------------------- 
    % Condiciones necesarias de KKT
    H=[Q*x+A'*lambda-F'*mu+d;A*x-b; -F*x+z+c;mu.*z];
       
    norma = norm(H, "inf");
    fprintf('%3.0f  %2.8f  %2.8f %2.8f %2.16f\n',cinter,norma,alfa,2*tau,condK)
end

end
