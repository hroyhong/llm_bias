

rm(list=ls())
library(dplyr)
library(DEoptim)

dat<-read.csv('/Users/hroyhong/Desktop/llm0412/task2/data/combined_complete.csv')


pa<-function(li){
  library(dplyr)
  library(DEoptim)
  
  dat<-read.csv('/Users/hroyhong/Desktop/llm0412/task2/data/combined_complete.csv')
  list_ID<-unique(dat$subj)
  subj<-length(unique(dat$subj))
  dd <- dat[dat$subj == list_ID[li] & (dat$casino == 2 | dat$casino == 3), ]
  
  LogL<-function(pp){
    # With this:
    alpha_pos_c <- pp[1] # Alpha for Positive PE Chosen
    alpha_neg_c <- pp[2] # Alpha for Negative PE Chosen
    alpha_pos_u <- pp[3] # Alpha for Positive PE Unchosen
    alpha_neg_u <- pp[4] # Alpha for Negative PE Unchosen
    tau <- pp[5]         # Inverse temperature
    logl<-0
    
    value2<-c(0,0) 
    value3<-c(0,0) 
    
    for (i in 1:nrow(dd)){
      choose=dd$decision[i] ## 1.left,2.right
      reward1=dd$reward[i] ## chosen option
      casino=dd$casino[i]
      
      
      if (casino==2){
        v1=exp(tau*value2[choose])
        vt<-exp(tau*value2[1])+exp(tau*value2[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value2[choose]  ## chosen option PE
        if(pe1>0){
          value2[choose]<-value2[choose] + alpha_pos_c * pe1 ## Chosen PE > 0
        }
        if(pe1<0){
          value2[choose]<-value2[choose] + alpha_neg_c * pe1 ## Chosen PE < 0
        }
        
        unchosen_index <- 3 - choose # Get index of the unchosen option (1 or 2)
        reward_unchosen <- dd$counterfactual_reward[i] # Use the correct column name
        pe_unchosen <- reward_unchosen - value2[unchosen_index] # PE for unchosen
        
        if (pe_unchosen > 0) {
          value2[unchosen_index] <- value2[unchosen_index] + alpha_pos_u * pe_unchosen # Unchosen PE > 0
        }
        if (pe_unchosen < 0) {
          value2[unchosen_index] <- value2[unchosen_index] + alpha_neg_u * pe_unchosen # Unchosen PE < 0
        }
        
      }
      
      if (casino==3){
        v1=exp(tau*value3[choose])
        vt<-exp(tau*value3[1])+exp(tau*value3[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value3[choose]  ## chosen option PE
        if(pe1>0){
          value3[choose]<-value3[choose] + alpha_pos_c * pe1 ## Chosen PE > 0
        }
        if(pe1<0){
          value3[choose]<-value3[choose] + alpha_neg_c * pe1 ## Chosen PE < 0
        }
        
        # --- Add this for unchosen update ---
        unchosen_index <- 3 - choose # Get index of the unchosen option (1 or 2)
        reward_unchosen <- dd$counterfactual_reward[i] # Use the correct column name
        pe_unchosen <- reward_unchosen - value3[unchosen_index] # PE for unchosen
        
        if (pe_unchosen > 0) {
          value3[unchosen_index] <- value3[unchosen_index] + alpha_pos_u * pe_unchosen # Unchosen PE > 0
        }
        if (pe_unchosen < 0) {
          value3[unchosen_index] <- value3[unchosen_index] + alpha_neg_u * pe_unchosen # Unchosen PE < 0
        }
      }
      
    }
    logl<-(-logl)
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0,0,0),upper=c(1,1,1,1,40),control = DEoptim.control(trace = FALSE,NP = 80,itermax = 250))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1] # alpha_pos_c
  b<-dd1$optim$bestmem[2] # alpha_neg_c
  c<-dd1$optim$bestmem[3] # alpha_pos_u
  d<-dd1$optim$bestmem[4] # alpha_neg_u
  e<-dd1$optim$bestmem[5] # tau
  f<-dd1$optim$bestval   # LogLikelihood
  g<-list_ID[li]         # Subject ID
  final=c(a,b,c,d,e,f,g)
  return(final)
}


##################


library(parallel)
library(snow)
library(SnowballC)

list_ID<-unique(dat$subj)

cl <- makeCluster(8)

results<-parLapply(cl,1:length(list_ID),pa)
res<-do.call('rbind',results)

write.table(res,"/Users/hroyhong/Desktop/llm0412/task2/result/parallel_mle_result_complete.csv",row.names=FALSE,col.names=TRUE,sep=",")