using LinearAlgebra
T = Float64

function trace(matrix)
    tr = sum(diag(matrix))
    return tr
end

function Q_p(A, p)
    Qp = Matrix{T}(I, size(A))
    for k in 1:p
        ck = factorial(2*p-k)*factorial(p)/(factorial(2*p)*factorial(p-k)*factorial(k))
        Qp = Qp + ck * A^k
    end
    return Qp
end


# function expm(A, p)
#     λ = trace(A)/size(A)[1]
#     B = A - λ*Matrix{T}(I, size(A))
#     # Diagonalize the matrix
#     D, P = eigen(B)

#     norm1 = norm(B, 1)
#     if norm1 >=1
#         m = ceil(Int, log2(norm1))
#     else 
#         m = 0
#     end
#     B = 2. ^(-m) * B
#     expB = (Q_p(-B, p)^(-1)) * Q_p(B, p)
#     expB = expB ^(2. ^m)
#     expm = exp(λ) * P *  Diagonal(D) * expB * Diagonal(D)^(-1) * P^-1
#     return expm
# end
A = [1.0 2.0; 3.0 4.0]

expm(A, 6)

function compute_pade_coefficients(p, q)
    c = zeros(p + q + 1)
    c[p + 1] = 1.0
    
    for j in 1:p
        c[p + 1 - j] = c[p + 2 - j] * ((p + q + 1 - j) / j)
    end
    
    d = zeros(p + q + 1)
    d[q + 1] = 1.0
    
    for j in 1:q
        d[q + 1 - j] = d[q + 2 - j] * (-((p + q + 1 - j) / j))
    end
    
    return c, d
end

function Balancing_Osborne(matrix)
    rows, columns = size(matrix)
    D = Matrix{T}(I, rows, columns)
    for k in 1:10
        for i in 1:r
            original_vector = matrix[:,i]
            x = vcat(original_vector[1:i - 1], original_vector[i + 1:end])
            c = norm(x, 2)

            original_vector = matrix[i,:]
            x = vcat(original_vector[1:i - 1], original_vector[i + 1:end])
            r = norm(x, 2)

            f = sqrt(r/c)

            D[i,i] = f * D[i,i]

            matrix[:,i] = matrix[:,i] * f
            matrix[i,:] = matrix[i,:] / f
        end
    end
    return D, matrix
end

function Balancing(matrix, p)
    rows, columns = size(matrix)
    D = Matrix{T}(I, rows, columns)
    converged = 0
    while converged == 0
        converged = 1
        for i in 1:rows
            c = norm(matrix[:,i], p)
            r = norm(matrix[i,:], p)
            s = c^p + r^p
            f = 1
            β = 10
            while c < r/β
                c = c*β
                r = r/β
                f = f * β
            end
            while c>= r*β
                c = c/β
                r = r*β
                f = f/β
            end
            if c^p + r^p < 0.95*s
                converged = 0
                D[i,i] = f * D[i,i]

                matrix[:,i] = matrix[:,i] * f
                matrix[i,:] = matrix[i,:] / f
            end
        end
    end
    return D, matrix
end

function expm(A)
    # θ_13 = 5.371920351148152
    θ_13 = 0.001
    μ = trace(A)/size(A)[1]
    A = A - μ * Matrix{T}(I, size(A))
    # D, A = Balancing(A, 2)
    # D, A = Balancing_Osborne(A)
    norm1 = norm(A, 1)
    if norm1 >=θ_13
        s = ceil(Int, log2(norm1/θ_13))
    else 
        s = 0
    end
    A = A/(2. ^s)
    println(s)
    A_2 = A^2
    A_4 = A_2^2
    A_6 = A_2 * A_4

    b = [64764752532480000, 32382376266240000, 7771770303897600,
        31187353796428800, 129060195264000, 10559470521600,
        4670442572800, 33522128640, 1323241920,
        540840800, 960960, 16380, 182, 1]

    U = A*(A_6*(b[14]*A_6 + b[12]*A_4 + b[10]*A_2) + b[8]*A_6 + b[6]*A_4 + b[4]*A_2 + b[2]*Matrix{T}(I, size(A)))
    V = A_6*(b[13]*A_6 + b[11]*A_4 + b[9]*A_2) + b[7]*A_6 + b[5]*A_4 + b[3]*A_2 + b[1]*Matrix{T}(I, size(A))

    r_13 = (-U+V)^(-1)*(U+V)
    r_13 = r_13^(2. ^s)
    # return exp(μ) * D * r_13 * D^(-1) 
    return exp(μ) * r_13
end