#------------Relative Model 初始值为 0-------------------------
##------------Relative Model: Self_Reward-------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Self',]
data<-data[data$condition=='Reward',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Self',]
  data<-data[data$condition=='Reward',]
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    len1<-pp[1]
    len2=pp[2]
    tau<-pp[3]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(0,0,0) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[3] = value [3] + len2 * pe2
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 200,itermax = 1000))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  #d<-dd1$optim$bestmem[4]
  c_val<-dd1$optim$bestval
  d<-'self_reward'
  e<-list_ID[li]
  final=c(a,b,c,c_val,d,e)
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_self_reward.csv",row.names=FALSE,col.names=TRUE,sep=",")

#------------Relative Model 初始值为 0.5-------------------------
##------------Relative Model: Self_Reward-------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Self',]
data<-data[data$condition=='Reward',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Self',]
  data<-data[data$condition=='Reward',]
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    len1<-pp[1]
    len2=pp[2]
    tau<-pp[3]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(0.5,0.5,0.5) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 200,itermax = 1000))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  #d<-dd1$optim$bestmem[4]
  c_val<-dd1$optim$bestval
  d<-'self_reward'
  e<-list_ID[li]
  final=c(a,b,c,c_val,d,e)
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_self_reward.csv",row.names=FALSE,col.names=TRUE,sep=",")




##------------Relative Model: Self_Punishment-------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Self',]
data<-data[data$condition=='Punishment',]

pa<-function(li){
  
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Self',]
  data<-data[data$condition=='Punishment',]
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    len1<-pp[1]
    len2=pp[2]
    tau<-pp[3]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(-0.5,-0.5,-0.5) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 200,itermax = 1000))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  #d<-dd1$optim$bestmem[4]
  c_val<-dd1$optim$bestval
  d<-'self_punishment'
  e<-list_ID[li]
  final=c(a,b,c,c_val,d,e)
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_self_punishment.csv",row.names=FALSE,col.names=TRUE,sep=",")



##------------Relative Model: Other_Reward-------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Other',]
data<-data[data$condition=='Reward',]

pa<-function(li){
  
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Other',]
  data<-data[data$condition=='Reward',]
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    len1<-pp[1]
    len2=pp[2]
    tau<-pp[3]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(0.5,0.5,0.5) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 200,itermax = 1000))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  #d<-dd1$optim$bestmem[4]
  c_val<-dd1$optim$bestval
  d<-'other_reward'
  e<-list_ID[li]
  final=c(a,b,c,c_val,d,e)
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_other_reward.csv",row.names=FALSE,col.names=TRUE,sep=",")



##------------Relative Model: Other_Punishment-------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Other',]
data<-data[data$condition=='Punishment',]

pa<-function(li){
  
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Other',]
  data<-data[data$condition=='Punishment',]
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    len1<-pp[1]
    len2=pp[2]
    tau<-pp[3]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(-0.5,-0.5,-0.5) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 200,itermax = 1000))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  #d<-dd1$optim$bestmem[4]
  c_val<-dd1$optim$bestval
  d<-'other_punishment'
  e<-list_ID[li]
  final=c(a,b,c,c_val,d,e)
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_other_punishment.csv",row.names=FALSE,col.names=TRUE,sep=",")






relative_model_self_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_self_reward.csv')
relative_model_self_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_self_punishment.csv')
relative_model_other_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_other_reward.csv')
relative_model_other_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/2relative_model_other_punishment.csv')

relative_model=rbind(relative_model_self_reward, relative_model_self_punishment, relative_model_other_reward, relative_model_other_punishment)

mean_len1=round(mean(relative_model$len1),3); se_len1=round(sd(relative_model$len1)/sqrt(length(relative_model$len1)),3)
mean_len2=round(mean(relative_model$len2),3);se_len2=round(sd(relative_model$len2)/sqrt(length(relative_model$len2)),3)
mean_tau=round(mean(relative_model$tau),3);se_tau=round(sd(relative_model$tau)/sqrt(length(relative_model$tau)),3)
mean_ll=round(mean(relative_model$LL),3);se_ll=round(sd(relative_model$LL)/sqrt(length(relative_model$LL)),3)

relative_model$loglik = - relative_model$LL
p2=3
relative_model$AIC_values <- -2 * relative_model$loglik + 2 * p2
relative_model_AIC=mean(relative_model$AIC_values)
se_AIC=round(sd(relative_model$AIC_values)/sqrt(length(relative_model$AIC_values)),3)

n=24
relative_model$BIC_values <- -2 * relative_model$loglik + log(n) * p2
relative_model_BIC=mean(relative_model$BIC_values)
se_BIC=round(sd(relative_model$BIC_values)/sqrt(length(relative_model$BIC_values)),3)
















































##------------Relative Model: self-reward 初始值为x-------------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Self',]
data<-data[data$condition=='Reward',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Self',]
  data<-data[data$condition=='Reward',]

  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    x<-pp[1]
    len1<-pp[2]
    len2=pp[3]
    tau<-pp[4]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(x,x,x) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0,0),upper=c(1,1,1,20),control = DEoptim.control(trace = FALSE,NP = 100,itermax = 500))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2] # option learning rate
  c<-dd1$optim$bestmem[3] # context learning rate
  d<-dd1$optim$bestmem[4] # tau
  #d<-dd1$optim$bestmem[4]
  e<-dd1$optim$bestval
  f<-'Self_Reward_Qx'
  g<-list_ID[li]
  
  final=c(a,b,c,d,e,f,g)
  
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)



res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('x','len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_reward_Qx.csv",row.names=FALSE,col.names=TRUE,sep=",")



##------------Relative Model: self-punishment 初始值为x-------------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Self',]
data<-data[data$condition=='Punishment',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Self',]
  data<-data[data$condition=='Punishment',]
  
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    x<-pp[1]
    len1<-pp[2]
    len2=pp[3]
    tau<-pp[4]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(x,x,x) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(-1,0,0,0),upper=c(0,1,1,20),control = DEoptim.control(trace = FALSE,NP = 100,itermax = 500))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2] # option learning rate
  c<-dd1$optim$bestmem[3] # context learning rate
  d<-dd1$optim$bestmem[4] # tau
  #d<-dd1$optim$bestmem[4]
  e<-dd1$optim$bestval
  f<-'Self_Punishment_Qx'
  g<-list_ID[li]
  
  final=c(a,b,c,d,e,f,g)
  
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)



res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('x','len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_punishment_Qx.csv",row.names=FALSE,col.names=TRUE,sep=",")


##------------Relative Model: other-reward 初始值为x-------------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Other',]
data<-data[data$condition=='Reward',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Other',]
  data<-data[data$condition=='Reward',]
  
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    x<-pp[1]
    len1<-pp[2]
    len2=pp[3]
    tau<-pp[4]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(x,x,x) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0,0),upper=c(1,1,1,20),control = DEoptim.control(trace = FALSE,NP = 100,itermax = 500))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2] # option learning rate
  c<-dd1$optim$bestmem[3] # context learning rate
  d<-dd1$optim$bestmem[4] # tau
  #d<-dd1$optim$bestmem[4]
  e<-dd1$optim$bestval
  f<-'Other_Reward_Qx'
  g<-list_ID[li]
  
  final=c(a,b,c,d,e,f,g)
  
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)



res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('x','len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_reward_Qx.csv",row.names=FALSE,col.names=TRUE,sep=",")



##------------Relative Model: other-punishment 初始值为x-------------------------------
path='/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour'

dat<-read.csv(file.path(path, 'combined.csv'))
data<-dat[dat$object=='Other',]
data<-data[data$condition=='Punishment',]

pa<-function(li){
  library(DEoptim)
  blocks=3
  trials=24
  dat<-read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/combined.csv')
  data<-dat[dat$object=='Other',]
  data<-data[data$condition=='Punishment',]
  
  data_fun<-data
  list_ID<-unique(data_fun$ID)
  subj<-length(unique(data_fun$ID))
  check<-sort(unique(data_fun$pic1))
  dd<-data_fun[data_fun$ID==list_ID[li],]
  LogL<-function(pp){
    x<-pp[1]
    len1<-pp[2]
    len2=pp[3]
    tau<-pp[4]
    logl<-0
    for (m in 1:blocks){  ##3 conditions X 2 repeats by the number of pic1
      data_real<-dd[dd$pic1==check[m],]
      value<-c(x,x,x) 
      for (i in 1:trials){
        choose=data_real$decisions[i] ## 1.left,2.right
        reward1=data_real$outcome1[i] ## chosen option
        #reward2=data_real$outcome2[i] ## unchosen option
        v1=exp(tau*value[choose])
        vt<-exp(tau*value[1])+exp(tau*value[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value[3]-value[choose]  ## chosen option
        pe2=(reward1+value[(3-choose)])/2-value[3]
        value[choose]<-value[choose]+len1*pe1 ## PE>0
        value[3] = value [3] + len2 * pe2
        
      }
    }
    logl<-(-logl)
    if (is.finite(logl) == F) {logl=100000;}
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(-1,0,0,0),upper=c(0,1,1,20),control = DEoptim.control(trace = FALSE,NP = 100,itermax = 500))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2] # option learning rate
  c<-dd1$optim$bestmem[3] # context learning rate
  d<-dd1$optim$bestmem[4] # tau
  #d<-dd1$optim$bestmem[4]
  e<-dd1$optim$bestval
  f<-'Other_Punishment_Qx'
  g<-list_ID[li]
  
  final=c(a,b,c,d,e,f,g)
  
  return(final)
}

##################
library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(data$ID)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)



res_df_school<-do.call('rbind',results)

colnames(res_df_school) = c('x','len1','len2','tau','LL','condition','ID')

write.table(res_df_school,"/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_punishment_Qx.csv",row.names=FALSE,col.names=TRUE,sep=",")


#-----relative model------
relative_model2_self_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_reward_Qx.csv')
relative_model2_self_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_punishment_Qx.csv')
relative_model2_other_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_reward_Qx.csv')
relative_model2_other_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_punishment_Qx.csv')


# relative_model2_self_reward=read.csv('C:/Users/17286/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_reward_Qx.csv')
# relative_model2_self_punishment=read.csv('C:/Users/17286/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_self_punishment_Qx.csv')
# relative_model2_other_reward=read.csv('C:/Users/17286/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_reward_Qx.csv')
# relative_model2_other_punishment=read.csv('C:/Users/17286/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/3relative_model_other_punishment_Qx.csv')

relative_model2=rbind(relative_model2_self_reward,relative_model2_self_punishment,relative_model2_other_reward,relative_model2_other_punishment)

# install.packages('openxlsx')
library(openxlsx)
write.xlsx(relative_model2_2, file='C:/Users/17286/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/relative_model2.xlsx')


relative_model2_2=relative_model2 %>%
  group_by(condition) %>%
  summarize(mean_x=round(mean(x),3), se_x=round(sd(x)/sqrt(n()),3),
            mean_len1=round(mean(len1),3), se_len1=round(sd(len1)/sqrt(n()),3),
            mean_len2=round(mean(len2),3), se_len2=round(sd(len2)/sqrt(n()),3),
            mean_tau=round(mean(tau),3), se_tau=round(sd(tau)/sqrt(n()),3),
            mean_LL=round(mean(LL),3), se_LL=round(sd(LL)/sqrt(n()),3))

relative_model2_3=relative_model2 %>%
  summarize(mean_x=round(mean(x),3), se_x=round(sd(x)/sqrt(n()),3),
            mean_len1=round(mean(len1),3), se_len1=round(sd(len1)/sqrt(n()),3),
            mean_len2=round(mean(len2),3), se_len2=round(sd(len2)/sqrt(n()),3),
            mean_tau=round(mean(tau),3), se_tau=round(sd(tau)/sqrt(n()),3),
            mean_LL=round(mean(LL),3), se_LL=round(sd(LL)/sqrt(n()),3))

relative_model2_4=relative_model2 %>%
  group_by(ID) %>%
  summarize(mean_x=round(mean(x),3), se_x=round(sd(x)/sqrt(n()),3),
            mean_len1=round(mean(len1),3), se_len1=round(sd(len1)/sqrt(n()),3),
            mean_len2=round(mean(len2),3), se_len2=round(sd(len2)/sqrt(n()),3),
            mean_tau=round(mean(tau),3), se_tau=round(sd(tau)/sqrt(n()),3),
            mean_LL=round(mean(LL),3), se_LL=round(sd(LL)/sqrt(n()),3))

##计算relative model的LL, AIC，BIC
relative_model2$loglik = - relative_model2$LL
p2=4
relative_model2$AIC_values <- -2 * relative_model2$loglik + 2 * p2
relative_model2_AIC=mean(relative_model2$AIC_values)
se_AIC=round(sd(relative_model2$AIC_values)/sqrt(length(relative_model2$AIC_values)),3)

n=24
relative_model2$BIC_values <- -2 * relative_model2$loglik + log(n) * p2
relative_model2_BIC=mean(relative_model2$BIC_values)
se_BIC=round(sd(relative_model2$BIC_values)/sqrt(length(relative_model2$BIC_values)),3)



#-------------------------model comparison----------------------------------------
abs_self_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/self_reward.csv')
colnames(abs_self_reward)=c('len1', 'tau', 'LL', 'condition', 'ID')
abs_self_reward$condition='abs_self_reward'
abs_self_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/self_punishment.csv')
colnames(abs_self_punishment)=c('len1', 'tau', 'LL', 'condition', 'ID')
abs_self_punishment$condition='abs_self_punishment'
abs_other_reward=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/other_reward.csv')
colnames(abs_other_reward)=c('len1', 'tau', 'LL', 'condition', 'ID')
abs_other_reward$condition='abs_other_reward'
abs_other_punishment=read.csv('/Users/jasminewu/Dropbox/Desktop_syn/2.self_other/self_other1/behaviour/other_punishment.csv')
colnames(abs_other_punishment)=c('len1', 'tau', 'LL', 'condition', 'ID')
abs_other_punishment$condition='abs_other_punishment'

abs_model=rbind(abs_self_reward, abs_self_punishment, abs_other_reward, abs_other_punishment)

abs_model2= abs_model %>%
  group_by(ID) %>%
  summarize(mean_len1=round(mean(len1),3), se_len1=round(sd(len1)/sqrt(n()),3),
            mean_tau=round(mean(tau),3), se_tau=round(sd(tau)/sqrt(n()),3),
            mean_LL=round(mean(LL),3), se_LL=round(sd(LL)/sqrt(n()),3))

#abs_model的42个LL值，越大代表模型拟合越好
-abs_model2$mean_LL
#relative_model的42个LL值，越大代表模型拟合越好
-relative_model2_4$mean_LL

#bmsR包比较
m <- structure(c(-abs_model2$mean_LL, -relative_model2_4$mean_LL), .Dim = c(42L, 2L), 
               .Dimnames = list(c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", '13',
                                  '14','15','16','17','18','19','20','21','22','23','24','25','26',
                                  '27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42'), 
                                c("abs_model", "relative_model")))
bms0 <- VB_bms(m)
names(bms0)
bms0$alpha
bms0$r
bms0$xp
bms0$bor
bms0$pxp


ll <- data.frame(
  Model = c("ABS", "REL"),
  XP = c(0.05, 1) # 将0替换为一个很小的值，比如0.001，这样柱形仍有一定高度
)

# 生成柱状图
ggplot(ll, aes(x = Model, y = XP)) +
  geom_bar(stat = "identity", fill = "gray") +
  ylim(0, 1) + # 设置y轴的范围
  geom_hline(yintercept = 0.95,linetype = "dashed")+
  labs(y = "Exceedance probability (XP)", x = "", title='BMS') +
  theme_bw() +
  theme(
    axis.title.y = element_text(size = 12),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 10),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title=element_text(hjust=0.5)
  )
  
geom_hline()



