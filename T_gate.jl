using Random
using LinearAlgebra

K = 2 * pi * 12.5 * 10 ^ 6
xi = 1.5 * pi
alpha = 0.5
alpha_im = 0.5 * exp(1.0im * xi)
epsilon_2 = alpha ^ 2 * K
N_p = 2.0 * (1.0 + exp(- 2.0 * alpha^2))
N_m = 2.0 * (1.0 - exp(- 2.0 * alpha^2))
T = 10^(-6)
D = 2
f = 7
a = zeros(ComplexF64 , f+1 , f+1)
a_dagger = zeros(ComplexF64 , f+1 , f+1)

U_G = Matrix{ComplexF64}(undef , 2 , 2)
U_G = [1 0 ; 0 exp(1.0im * pi / 4.0)]
    
I = zeros(ComplexF64 , f , f)

for i in 1 : f
    I[i , i] = 1.0 + 0.0im
end

lr1 = 0.000001
lr2 = 0.00001


for i in 1 : f
    a[i , i+1] = sqrt(i)
    a_dagger[i+1 , i] = sqrt(i)
end

hbar = 1.0

num = 3
eta0 = 0
mu0 = 0

omega1 = 1
omega2 = 1

W1 = Vector{Float64}(undef , num)
W2 = Vector{Float64}(undef , num)
B1 = Vector{Float64}(undef , num)
phi1 = Vector{Float64}(undef , num)

W3 = Vector{Float64}(undef , num)
W4 = Vector{Float64}(undef , num)
B3 = Vector{Float64}(undef , num)
phi2 = Vector{Float64}(undef , num) 


W1 = [0.3 ; -0.9 ; 0.4]
W2 = [1.2 ; 0.5 ; 0.7]
B1 = [0.5 ; 1.7 ; 1.3]
phi1 =[3.1 ; 1.7 ; 1.3]

W3 = [1 ; -2.7 ; 1.7]
W4 = [4.2 ; 0.1 ; 0]
B3 = [-0.1 ; 1 ; -1.7]
phi2 = [3.1 ; 2.7 ; 1.9]

a1_dagger = a_dagger[1 : f , 1 : f]
a1 = a[1 : f , 1 : f]

A_4 = (a_dagger * a_dagger) * (a * a)
A_2 = a_dagger * a

num1 = A_2[1 : f , 1 : f]
a14 = A_4[1 : f , 1 : f]


function tanh_der(x)
    return 1.0 .- tanh.(x).^2
end

function mu_v(tau , W1 , B1 , phi1)
    C = zeros(ComplexF64 , num)
    C = cos.(omega1 * tau .+ phi1) .* W1
    return tanh.(C .+ B1)
end

function eta_v(tau , W3 , B3 , phi2)
    C = zeros(ComplexF64 , num)
    C = cos.(omega2 * tau .+ phi2) .* W3
    
    return tanh.(C .+ B3)
end

function mu(tau , W1 , W2 , B1 , phi1)
    C = dot(W2 , mu_v(tau , W1 , B1 , phi1)) + mu0 - dot(W2 , mu_v(0 , W1 , B1 , phi1))
    return C
end

function eta(tau , W3 , W4 , B3 , phi2)
    return dot(W4 , eta_v(tau , W3 , B3 , phi2)) + eta0 - dot(W4 , eta_v(0 , W3 , B3 , phi2))
end

function mu_dtau(tau , W1 , W2 , B1 , phi1) #对tau_1求导
    D = 0
    for i in 1 : num
        D = D + W2[i] * W1[i] * tanh_der(W1[i] * cos(omega1 * tau + phi1[i]) + B1[i]) * (- omega1 * sin(omega1 * tau + phi1[i]))
    end

    return real(D)
end

function mu_dt(tau , W1 , W2 , B1 , phi1) #对t求导
    return ((2.0 * pi) / T) * mu_dtau(tau , W1 , W2 , B1 , phi1)
end

function eta_dtau(tau , W3 , W4 , B3 , phi2) #对tau_2求导
    D = 0
    for i in 1 : num
        D = D + W4[i] * W3[i] * tanh_der(W3[i] * cos(omega2 * tau + phi2[i]) + B3[i]) * (- omega2 * sin(omega2 * tau + phi2[i]))
    end

    return real(D)
end

function eta_dt(tau , W3 , W4 , B3 , phi2) #对t求导
    return (pi / T) * eta_dtau(tau , W3 , W4 , B3 , phi2)
end

function Omega_x(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    C = 0.25 * (pi / T) * (eta_dtau(tau2 , W3 , W4 , B3 , phi2) * sin(eta(tau2 , W3 , W4 , B3 , phi2)) * sin(2.0 * mu(tau1 , W1 , W2 , B1 , phi1)) - 4 * mu_dtau(tau1 , W1 , W2 , B1 , phi1) * cos(eta(tau2 , W3 , W4 , B3 , phi2)))
    return C
end


function Omega_y(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    C = 0.25 * (pi / T) * (eta_dtau(tau2 , W3 , W4 , B3 , phi2) * cos(eta(tau2 , W3 , W4 , B3 , phi2)) * sin(2.0 * mu(tau1 , W1 , W2 , B1 , phi1)) + 4 * mu_dtau(tau1 , W1 , W2 , B1 , phi1) * sin(eta(tau2 , W3 , W4 , B3 , phi2)))
    return C
end

function Omega_z(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    C =  - 0.5 * (pi / T) * eta_dtau(tau2 , W3 , W4 , B3  , phi2) * (sin(mu(tau1 , W1 , W2 , B1 , phi1)))^2
    return C
end
 
function epsilon_re(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    C =  sqrt(N_p * N_m) * (Omega_x(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) * cos(xi) - exp(2.0 * alpha^2) * Omega_y(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) * sin(xi)) / (4.0 * alpha)
    return C
end

function epsilon_im(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    return sqrt(N_p * N_m) * (Omega_x(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) * sin(xi) + exp(2.0 * alpha^2) * Omega_y(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) * cos(xi)) / (4.0 * alpha)
end

function chi(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    C = (- 2.0 * Omega_z(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)* N_p * N_m) / ((N_p^2 - N_m^2) * alpha^2 )
    return C
end

H_cat = zeros(ComplexF64 , f , f)
H_cat = - K .* a14 .+ epsilon_2 .* (exp(2.0im * xi) .* a1_dagger * a1_dagger + exp(-2.0im * xi) .* a1 * a1)

function H_c(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)
    return chi(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) .* num1 .+ (epsilon_re(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) + 1.0im * epsilon_im(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)) .* a1_dagger .+ (epsilon_re(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2) - 1.0im * epsilon_im(tau1 , W1 , W2 , B1 , phi1 , tau2 , W3 , W4 , B3 , phi2)) .* a1
end

function factorial(n)
    num = 1
    if n == 0
        return 1
    else
        for i in 2 : n
            num = num * i
        end
        
        return num
    end
end

state_p = zeros(ComplexF64 , f)
state_m = zeros(ComplexF64 , f)
c_p = zeros(ComplexF64 , f)
c_m = zeros(ComplexF64 , f)

for i in 1 : f
    state_p[i] = alpha^(i - 1) * exp(1.0im * (i - 1) * xi) / sqrt(factorial(i - 1))
    state_m[i] = (- alpha)^(i - 1) * exp(1.0im * (i - 1) * xi) / sqrt(factorial(i - 1))
end

state_p = state_p .* exp(- alpha^2 * 0.5)
state_m = state_m .* exp(- alpha^2 * 0.5)

c_p = 1.0 / sqrt(N_p) .* (state_p .+ state_m)
c_m = 1.0 / sqrt(N_m) .* (state_p .- state_m)

function projective(U)
    M1 = zeros(ComplexF64 , 2 , 2)
    M1[1 , 1] = dot(c_p , U * c_p)
    M1[1 , 2] = dot(c_p , U * c_m)
    M1[2 , 1] = dot(c_m , U * c_p)
    M1[2 , 2] = dot(c_m , U * c_m)
    return M1
end

function M_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    return -1.0im * T / (2.0 * pi * mu_dtau(tau , W1 , W2 , B1 , phi1)) .* U_G' * projective(H * U)
end

function M_dagger_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    return 1.0im * T / (2.0 * pi * mu_dtau(tau , W1 , W2 , B1 , phi1)) .*  projective(U' * H') * U_G
end

function M_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    return -1.0im * T / (pi * eta_dtau(tau , W3 , W4 , B3 , phi2)) * U_G' * projective(H * U)
end

function M_dagger_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    return 1.0im * T / (pi * eta_dtau(tau , W3 , W4 , B3 , phi2)) * projective(U' * H') * U_G
end

function f_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    M = U_G' * projective(U)
    C1 = tr(M_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G) * M') + tr(M * M_dagger_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G))
    k1 = tr(M)
    k2 = tr(M_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G))
    C2 = 2 * real(k1) * real(k2) + 2 * imag(k1) * imag(k2)
    return (C1 + C2) / (D * (D + 1))
end

function f_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    M = U_G' * projective(U)
    C1 = tr(M_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G) * M') + tr(M * M_dagger_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G))
    k1 = tr(M)
    k2 = tr(M_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G))
    C2 = 2 * real(k1) * real(k2) + 2 * imag(k1) * imag(k2)
    return (C1 + C2) / (D * (D + 1))
end

function f_dW2(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = mu_v(tau , W1 , B1 , phi1)
    E = f_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G) .* C
    return real(E)
end

function f_dW1(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1) .* cos.(omega1 * tau .+ phi1)
    
    E = f_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G) .* C
    return real(E)
end

function f_dB1(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1)

    E = f_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G) .* C
    return real(E)
end

function f_dphi1(tau , W1 , W2 , B1 , phi1 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1) .* W1 .* (- sin.(omega1 * tau .+ phi1))
    
    E = f_dmu(tau , W1 , W2 , B1 , phi1 , H , U , U_G) .* C
    return real(E)
end

function f_dW4(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = eta_v(tau , W3 , W4 , phi2)
    E = f_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G) .* C
    return real(E)
end
    
function f_dW3(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W4 .* tanh_der(W3 .* cos.(omega2 * tau .+ phi2) .+ B3) .* cos.(omega2 * tau .+ phi2)

    E = f_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G) .* C
    return real(E)
end
    
function f_dB3(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W4 .* tanh_der(W3 .* cos.(omega2 * tau .+ phi2) .+ B3)
    E = f_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G) .* C
    return real(E)
end
    
function f_dphi2(tau , W3 , W4 , B3 , phi2 , H , U , U_G)
    C = zeros(ComplexF64 , num)
    C = W4 .* tanh_der(W3 .* cos.(omega2 * tau .+ phi2) .+ B3) .* W3 .* (- sin.(omega2 * tau .+ phi2))
    E = f_deta(tau , W3 , W4 , B3 , phi2 , H , U , U_G) .* C
    return real(E)
end
    

function runge_kutta(y, x, dx, f)
    k1 = dx .* f(y, x)
    k2 = dx .* f(y .+ 0.5 * k1, x + 0.5 * dx)
    k3 = dx .* f(y .+ 0.5 * k2, x + 0.5 * dx)
    k4 = dx .* f(y .+ k3, x + dx)
    return y .+ (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4) ./ 6.0
end


function fun(y , t)
    global W1
    global B1
    global W2
    global phi1
    
    global W3
    global B3
    global W4
    global phi2

    H = zeros(ComplexF64 , f , f)
    t1 = 2.0 * pi * t / T
    t2 = pi * t / T
    H = H_c(t1 , W1 , W2 , B1 , phi1 , t2 , W3 , W4 , B3 , phi2)
    C = zeros(ComplexF64 , f , f)
    C = (H_cat + H) * y * (-1.0im) / hbar
    return C
end

function Backforward(t::Vector{Float64})::Float64
    global W1
    global B1
    global W2
    global phi1
    
    global W3
    global B3
    global W4
    global phi2

    tau1 = 2 * pi .* t ./ T
    tau2 = pi .* t ./ T
    dt = t[2] - t[1]


    U = I

    for i in 1 : length(t)
        W1 = W1 + lr1 * f_dW1(tau1[i] , W1 , W2 , B1 , phi1 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        B1 = B1 + lr1 * f_dB1(tau1[i] , W1 , W2 , B1 , phi1 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        W2 = W2 + lr2 * f_dW2(tau1[i] , W1 , W2 , B1 , phi1 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        phi1 = phi1 + lr1 * f_dphi1(tau1[i] , W1 , W2 , B1 , phi1 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)

        W3 = W3 + lr1 * f_dW3(tau2[i] , W3 , W4 , B3 , phi2 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        B3 = B3 + lr1 * f_dB3(tau2[i] , W3 , W4 , B3 , phi2 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        W4 = W4 + lr2 * f_dW4(tau2[i] , W3 , W4 , B3 , phi2 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)
        phi2 = phi2 + lr1 * f_dphi2(tau2[i] , W3 , W4 , B3 , phi2 , H_c(tau1[i] , W1 , W2 , B1 , phi1 , tau2[i] , W3 , W4 , B3 , phi2) , U , U_G)

        U = runge_kutta(U , t[i] , dt , fun)
    end
    
    U1 = zeros(ComplexF64 , f , f , length(t) + 1)
    U1[: , : , 1] = I

    for n in 1 : length(t)
        U1[: ,: , n+1] = runge_kutta(U1[: , : , n] , t[n] , dt , fun)
    end
    
    M = zeros(ComplexF64 , 2 , 2)
    M1 = zeros(ComplexF64 , 2 , 2)
    M2 = zeros(ComplexF64 , 2 , 2)

    M1 = U_G'
    M2 = projective(U1[: , : , length(t)])
    M = M1 * M2

    f_avg = (tr(M * M') + abs(tr(M))^2) / (D * (D + 1))

    return real(f_avg)
end

step = 1000
t = collect(range(0, length = step, stop = T))
ephos = 3130
f_avg = zeros(ephos)

for i in 1 : 1240
    f_avg[i] = Backforward(t)
    println(f_avg[i])
end

for i in 1 : 1890
    f_avg[i + 1240] = Backforward(t)
    println(f_avg[i])
end

println(W1)
println(W2)
println(W3)
println(W4)
println(B1)
println(B3)
println(phi1)
println(phi2)