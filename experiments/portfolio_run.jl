using CSV, DataFrames, DecisionTree, Distributions, Gurobi, JuMP, LightGraphs, Parameters, SparseArrays, Statistics, ArgParse
include("../oracles/portfolio_oracle.jl")
include("../solver/util.jl")
include("../solver/sgd.jl")
include("../solver/reformulation.jl")
include("../solver/random_forests_po.jl")
include("../solver/validation_set.jl")
include("../experiments/replication_functions.jl")

# Fixed parameter settings (these never change)
p_features = 5 # Dimensão p do vetor de features
num_assets = 50 # Dimensão d do vetor de decisão e do vetor de custos (o problema aqui maximiza o retorno dos ativos)
num_trials = 50 # Número de repetições independentes do experimento
n_test = 10000 # Tamanho do conjunto de testes
n_test = 1000 # Tamanho do conjunto de testes
num_factors = 4

num_lambda = 1 # Parâmetro de regularização
lambda_max = 10.0^(-6) # Parâmetro de regularização
lambda_min_ratio = 1 # Parâmetro de regularização
holdout_percent = 0.25 # Dados de validação
different_validation_losses = false # Parâmetro para mudar a perda do treinamento e da validação
data_type = :poly_kernel


# Fixed parameter sets (these are also the same for all experiments)
n_train_vec = [100; 1000] # Vetor com o tamanho dos dados de treinamento (é esperado que o SPO tenha mais vantagens em relação ao MSE com n pequeno)
# n_train_vec = [100] 
n_sigmoid_polydegree_vec = [1; 4; 8; 16] # Grau de não-linearidade da função geradora c(x) Grau 1 = relação linear entre 𝑥 x e 𝑐 c (cenário "fácil", onde um preditor linear é bem-especificado); graus maiores (4, 8, 16) = relação cada vez mais não-linear, testando o desempenho quando o preditor (linear, no caso do SPO+ ou MSE) está mal-especificado em relação à verdadeira função geradora.
# n_sigmoid_polydegree_vec = [1] 
noise_multiplier_tau_vec = [1; 2] # controla a magnitude do ruído aditivo/multiplicativo na geração de 𝑐 c a partir de 𝑐 ˉ ( 𝑥 ) c ˉ (x) — ou seja, o quão "ruidosa" é a relação entre features e custos reais. 𝜏 τ maior = mais incerteza irredutível, o que deveria dificultar a tarefa de predição-decisão para todos os métodos, mas de formas potencialmente diferentes entre SPO+ e MSE.
# noise_multiplier_tau_vec = [1] 

# Set this based on expt_number (40 total)
rng_seed = 2223

# Run experiment and get results
# Note that Gurobi enviornments are set within this function call
expt_results = portfolio_multiple_replications(rng_seed, num_trials,
    num_assets, num_factors, n_train_vec, n_test, p_features,
    n_sigmoid_polydegree_vec, noise_multiplier_tau_vec;
    num_lambda = num_lambda, lambda_max = lambda_max, lambda_min_ratio = lambda_min_ratio, holdout_percent = holdout_percent,
    different_validation_losses = different_validation_losses, data_type = data_type)

csv_string = "portfolio_results.csv"
CSV.write(csv_string, expt_results)
