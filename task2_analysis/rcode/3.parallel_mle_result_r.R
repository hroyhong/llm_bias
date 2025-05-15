

rm(list=ls())
library(dplyr)
library(DEoptim)

dat<-read.csv('/Users/hroyhong/Desktop/llm0412/task1/data/combined_r.csv')


pa<-function(li){
  library(dplyr)
  library(DEoptim)
 
  dat<-read.csv('/Users/hroyhong/Desktop/llm0412/task1/data/combined_r.csv')
  list_ID<-unique(dat$subj)
  subj<-length(unique(dat$subj))
  dd <- dat[dat$subj == list_ID[li] & (dat$casino == 2 | dat$casino == 3), ]
  
  LogL<-function(pp){
    len1<-pp[1]
    len2<-pp[2]
    tau<-pp[3]
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
        
        pe1=reward1-value2[choose]  ## chosen option
        if(pe1>0){
          value2[choose]<-value2[choose]+len1*pe1 ## PE>0
        } 
        if(pe1<0){
          value2[choose]<-value2[choose]+len2*pe1 ## PE<0
        }
      }
      
      if (casino==3){
        v1=exp(tau*value3[choose])
        vt<-exp(tau*value3[1])+exp(tau*value3[2]) 
        logl<-logl+log(v1/vt) ##likelihood function:p=v1/vt
        
        pe1=reward1-value3[choose]  ## chosen option
        if(pe1>0){
          value3[choose]<-value3[choose]+len1*pe1 ## PE>0
        } 
        if(pe1<0){
          value3[choose]<-value3[choose]+len2*pe1 ## PE<0
        }
      }
      
    }
    logl<-(-logl)
    return(logl)
  }
  
  estimates_school = DEoptim(LogL,lower =c(0,0,0),upper=c(1,1,20),control = DEoptim.control(trace = FALSE,NP = 80,itermax = 250))
  dd1<-summary(estimates_school)
  a<-dd1$optim$bestmem[1]
  b<-dd1$optim$bestmem[2]
  c<-dd1$optim$bestmem[3]
  d<-dd1$optim$bestval
  e<-list_ID[li]
  final=c(a,b,c,d,e)
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

write.table(res,"/Users/hroyhong/Desktop/llm0412/task1/result/parallel_mle_result_r.csv",row.names=FALSE,col.names=TRUE,sep=",")