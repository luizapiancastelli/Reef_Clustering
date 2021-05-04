read_mcmc_file = function(filename){
  main <- import_main()
  py <- import_builtins()
  pickle <- import("pickle", convert = FALSE)
  filehandler = py$open(filename ,'rb') 
  chains = pickle$load(filehandler)
  chains = py_to_r(chains)
  return(chains)
}

relabel_mcmc = function(chains){
  
  m = length(chains)
  chain_size = dim(chains[[1]][[1]])[3]
  k = dim(chains[[1]][[1]])[1]
  p = chains[[1]][[3]]
  n = nrow(p)
  
  #Bind chains
  i =1 
  while(i <=m){
    chain_i = chains[[i]]
    rho_i = chain_i[[1]]
    z_i = chain_i[[2]]
    alpha_i = chain_i[[3]]
    beta_i = chain_i[[4]]
    prob_matrix_i = chain_i[[7]]
    if(i==1){
      z_bind = z_i
      rho_bind = rho_i
      alpha_bind = alpha_i
      beta_bind = beta_i
      prob_matrix = prob_matrix_i
    } else{
      z_bind = rbind(z_bind, z_i)
      rho_bind = abind(rho_bind, rho_i, along =3)
      alpha_bind = c(alpha_bind, alpha_i)
      beta_bind = c(beta_bind, beta_i)
      prob_matrix = abind(prob_matrix, prob_matrix_i, along =1)
    }
    i = i+1
  }
  relabel_stephen = stephens(prob_matrix)
  permutations = relabel_stephen$permutations
  rho_chain = rho_bind; z_chain = z_bind
  mcmc.pars = to_mcmc_array(rho_chain, z_chain)
  reordered_mcmc<-permute.mcmc(mcmc.pars,relabel_stephen$permutations)
  reordered_mcmc = reordered_mcmc$output
  
  rho_relab = to_list(reordered_mcmc, rho_chain)
  z_relab = z_from_permutations(permutations, z_chain)
  
  chains_relab_list = recover_lists(chains, rho_relab, z_relab)
  all_rho_relab = get_rho_chain_multiple(chains_relab_list)
  
  return(list('rho' = all_rho_relab,
              'z' = z_relab))
  
}


get_rho_chain_multiple = function(chains_relab_list){
  
  k =dim(chains_relab_list[[1]][[1]])[1]
  m = length(chains_relab_list)
  i = 1 #chain indicator
  
  while(i <=m){
    rho_params = chains_relab_list[[i]][[1]]
    l = 1
    while(l <= k){
      rho_l = rho_params[l,,]
      rho_l = data.frame(t(rho_l))
      names(rho_l) = paste0("prop", seq(1, 4,1))
      rho_l = gather(rho_l, "prop", "rho", 1:4)
      rho_l$group = as.character(l)
      
      if(l ==1){
        rho_chain_df = rho_l
      } else {
        rho_chain_df = rbind(rho_chain_df, rho_l)
      }
      l = l +1
    }
    rho_chain_df$chain = i
    if(i == 1){
      rho_all_chains = rho_chain_df
    } else {
      rho_all_chains = rbind(rho_all_chains, rho_chain_df)
    }
    i = i +1
  }
  
  rho_all_chains$element = paste0(rho_all_chains$group,"-", rho_all_chains$prop,"-", rho_all_chains$chain)
  rho_all_chains$rho_li = paste0(rho_all_chains$group,"-", rho_all_chains$prop)
  rho_all_chains$chain = as.factor(rho_all_chains$chain)
  rho_all_chains$index = rowid(rho_all_chains$element)
  
  return(rho_all_chains)
}


cluster_elements = function(z,K) {
  tab = function(l,z) sum(z==l,na.rm=TRUE)
  return(sapply(seq(1,K,1),tab, z))
}

summary_z = function(z_relab, k){
  n=ncol(z_relab)
  
  z_count = apply(z_relab, 2, cluster_elements, K = k)
  z_count = t(z_count)
  z_prop = prop.table(z_count, 1)
  
  mode_post = sapply(seq(1,nrow(z_prop),1), function(x) which(z_prop[x,1:k]==max(z_prop[x,1:k])))
  mode_post = unlist(lapply(mode_post,"[", 1))
  
  z_prop = data.frame(z_prop)
  z_prop$maxpost = mode_post
  
  out = list('probability' = z_prop,
             'mode' = mode_post)
  return(out)
  
}

to_mcmc_array= function(rho_chain, z_chain){
  
  n_chain = nrow(z_chain)
  n = ncol(z_chain)
  k = nrow(rho_chain[,,1])
  cat = ncol(rho_chain[,,1])
  mcmc_array = array(NA, dim=c(n_chain, k, cat))
  
  j = 1
  while(j<= n_chain){
    rho_j = rho_chain[,,j]
    z_j = z_chain[j,]
    l = 1
    while(l <=k){
      rho_l = rho_j[l,]
      zl = as.numeric(z_j == l)
      mcmc_array[j,l,]<- rho_l
      l = l+1
    }
    j = j+1
  }
  return(mcmc_array)
}

get_rho_chain = function(reordered_mcmc){
  
  k = dim(reordered_mcmc)[2]
  cat = dim(reordered_mcmc)[3]
  
  l = 1
  while(l <= k){
    rho_l = reordered_mcmc[,l,]
    rho_l = data.frame(rho_l)
    names(rho_l) = paste0("prop", seq(1, cat,1))
    rho_l = gather(rho_l, "prop", "rho", 1:cat)
    rho_l$group = as.character(l)
    
    if(l ==1){
      rho_chain_df = rho_l
    } else {
      rho_chain_df = rbind(rho_chain_df, rho_l)
    }
    l = l +1
  }
  return(rho_chain_df)
}

to_list = function(reordered_mcmc, rho_chain){
  rho_relab = rho_chain
  chain_size = dim(reordered_mcmc)[1]
  j =1
  while(j<= chain_size){
    rho_j = reordered_mcmc[j,,]
    rho_relab[,,j] = rho_j
    j = j+1
  }
  return(rho_relab)
}

z_from_permutations = function(permutations, z_chain){
  
  library(plyr)
  chain_size = nrow(permutations)
  n = ncol(z_chain)
  k = ncol(permutations)
  
  z_relab = matrix(NA, nrow = chain_size, ncol = n)
  j=1
  while(j<= chain_size){
    z_j = z_chain[j,]
    perm_j = permutations[j,]
    z_j_relab = mapvalues(z_j, from = as.character(seq(1,k,1)), to = as.character(perm_j))
    z_j_relab = as.numeric(z_j_relab)
    z_relab[j,] = z_j_relab
    j= j+1
  }
  
  return(z_relab)
  
}

recover_lists = function(all_chains, rho_relab, z_relab){
  chains_relabeled = all_chains
  m = length(all_chains)
  chain_size = dim(chains_relabeled[[1]][[1]])[3]
  
  chunk <- function(x,n) split(x, factor(sort(rank(x)%%n)))
  indexes = chunk(seq(1, chain_size*m, 1),m)
  
  i = 1
  while(i <= m){
    ind = indexes[[i]]
    first = ind[1]
    last = ind[length(ind)]
    
    rho_i_relab = rho_relab[,,first:last]
    z_i_relab = z_relab[first:last,]
    
    chains_relabeled[[i]][[1]] = rho_i_relab
    chains_relabeled[[i]][[2]] = z_i_relab
    i = i+1
    
  }
  return(chains_relabeled)
}

plot_rho_posterior = function(results_relab){
  
  results_relab$rho$prop = as.factor(results_relab$rho$prop)
  levels(results_relab$rho$prop) = paste("Proportion", seq(1,length(levels(results_relab$rho$prop)),1))
  
  #Trace plot colored by chain
  trace = ggplot(results_relab$rho, aes(index, rho, group = as.factor(chain))) +
    geom_line(aes(colour = as.factor(chain)), linetype = "dashed", alpha = 0.7) +
    facet_wrap(group ~ prop, scales = "free", ncol = length(unique(results_relab$rho$prop)))+ theme_bw()+
    labs(x = 'Iteration', y= expression(rho),colour = 'Chain')
  
  #Density plot
  density = ggplot(results_relab$rho, aes(x=rho, fill =group)) + 
    geom_density(alpha = 0.7, adjust = 2.5)+ 
    theme_bw() + theme(plot.title = element_text(hjust = 0.5)) +
    facet_wrap(group ~ prop, scales = "free", ncol = length(unique(results_relab$rho$prop))) +
    theme(legend.position = "none")+labs(x = expression(rho), y = 'Group')
  
  return(list('trace' = trace,
              'density' = density))
}
