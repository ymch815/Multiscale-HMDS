#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# compile all models

import cmdstanpy as stan

ltz_model = stan.CmdStanModel(stan_file='./lorentz2.stan')
cm_model = stan.CmdStanModel(stan_file='./CM2.stan')

relax_model = stan.CmdStanModel(stan_file='./relax.stan')
transform_model = stan.CmdStanModel(stan_file='./transform.stan')
