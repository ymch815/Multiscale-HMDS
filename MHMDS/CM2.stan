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
        int<lower=0> N;        // number of points
        int<lower=0> D;
        array[N] vector[D] coords;   // x1-xd lorentzian coordinates of data
    }
    transformed data{
        vector[N] coord_ts;
        
        for(i in 1:N)
            coord_ts[i] = sqrt(1.0 + dot_self(coords[i]));
    }
    parameters {
        vector[D] CM;                // center of mass euclidean coordinates
    }
    transformed parameters {
        real CM_t;
        
        CM_t = sqrt(1.0 + dot_self(CM));
    }
    model {
        real dist;
       
        for(i in 1:N){
            dist = hyp(CM_t, coord_ts[i], CM, coords[i]);
    
            target += -square(dist);
        }
    }