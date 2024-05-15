functions {
    real hyp(real t1, real t2, vector E1, vector E2){
            real xi = t1*t2 - dot_product(E1, E2);
            if(xi > 1)
                return acosh(t1*t2 - dot_product(E1, E2));
            else
                return 0.0;
      }
}
data {
    int<lower=0> Ne;           // number of already embedded reference points
    int<lower=0> Nn;           // number of new points
    int<lower=0> D;            // Dimension of space
    real<lower=0.001> lambda;  // curvature of initial embedding
    array[Ne] vector[D] euc_emb;     // spacelike lorentzian coordinates of existing embedded points
    vector<lower=0.0>[Ne] sig_e; // previous fitted error
    matrix[Nn, Ne] deltaij_mutual;    // matrix of data distances between reference point and new point
    matrix[Nn, Nn] deltaij;    // matrix of data distances of new point
}
parameters {
    array[Nn] vector[D] euc_new;                // directions
    vector<lower=0.0>[Nn] sig_n;
}
transformed parameters {
    vector[Ne] time_e;
    vector[Nn] time_n;
    
    for (i in 1:Ne)
        time_e[i] = sqrt(1.0 + dot_self(euc_emb[i]));
    for (i in 1:Nn)
        time_n[i] = sqrt(1.0 + dot_self(euc_new[i]));
}
model {
    real dist; //
    real dist_mutual; //
    real seff;
    real seff_mutual; 

    for (i in 1:Nn)
        sig_n[i] ~ inv_gamma(2.0, 0.5);
    
    for(i in 1:Nn){
        for (j in i+1:Nn){
            if (deltaij[i,j] > 0.0){
                dist = hyp(time_n[i], time_n[j], euc_new[i], euc_new[j]);
		seff = sqrt(square(sig_n[i]) + square(sig_n[j]));

                deltaij[i,j] ~ normal(dist/lambda, seff);
            }
        }
    }
    for(i in 1:Nn){
        for (j in 1:Ne){
            if (deltaij_mutual[i,j] > 0.0){
                dist_mutual = hyp(time_n[i], time_e[j], euc_new[i], euc_emb[j]);
		seff_mutual = sqrt(square(sig_n[i]) + square(sig_e[j]));
                deltaij_mutual[i,j] ~ normal(dist_mutual/lambda, seff_mutual);
            }
        }
    }
}