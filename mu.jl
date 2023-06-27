using LinearAlgebra

num = 20
W1 = rand(Float64, num) * 1
B1 = rand(Float64, num) * 1
phi1 = rand(Float64, num) * 1
W2 = rand(Float64, num) * 1
lr = 0.00001
omega1 = 1
mu0 = 0

function tanh_der(x)
return 1.0 .- tanh.(x).^2
end

function mu_v(tau , W1 , B1 , phi1)
    C = cos.(omega1 * tau .+ phi1) .* W1
    return tanh.(C .+ B1)
end

function mu(tau , W1 , W2 , B1 , phi1)
    C = dot(W2 , mu_v(tau , W1 , B1 , phi1)) + mu0 - dot(W2 , mu_v(0 , W1 , B1 , phi1))
    return C
end

function dmu_dW2(tau , W1 , W2 , B1 , phi1)
    C = mu_v(tau , W1 , B1 , phi1)
    return C
end

function dmu_dW1(tau , W1 , W2 , B1 , phi1)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1) .* cos.(omega1 * tau .+ phi1)
    return C
end

function dmu_dphi1(tau , W1 , W2 , B1 , phi1)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1) .* W1 .* (- sin.(omega1 * tau .+ phi1))
    return C
end

function dmu_dB1(tau , W1 , W2 , B1 , phi1)
    C = W2 .* tanh_der(W1 .* cos.(omega1 * tau .+ phi1) .+ B1)
    return C
end

function backforward()
    x_train = range(0, stop=2π, length=100)
    y_train = (1 .- cos.(x_train)) *  + mu0
    
    for i in 1:100
        y_pred = mu(x_train[i] , W1 , W2 , B1 , phi1)
        L = y_pred - y_train[i]
        dL_dW1 = L * dmu_dW1(x_train[i] , W1 , W2 , B1 , phi1)
        dL_dW2 = L * dmu_dW2(x_train[i] , W1 , W2 , B1 , phi1)
        dL_dB1 = L * dmu_dB1(x_train[i] , W1 , W2 , B1 , phi1)
        dL_dphi1 = L * dmu_dphi1(x_train[i] , W1 , W2 , B1 , phi1)
        global W1 = W1 - lr * dL_dW1
        global W2 = W2 - lr * dL_dW2
        global B1 = B1 - lr * dL_dB1
        global phi1 = phi1 - lr * dL_dphi1
    end
end

x_train = range(0, stop=2π, length=100)
y_train = (1 .- cos.(x_train)) * pi + mu0
epcho = 1000000

for i in 1 : epcho
    backforward()
    L = 0
    for j in 1 : 100
        L = L + (mu(x_train[j] , W1 , W2 , B1 , phi1) - y_train[j])^2 # loss function
    end
    L = L / 100
end


