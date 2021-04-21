##' @description this create learning rate scheduler.
##'
##' @title Create learning rate scheduler.
##' @param lr_rate learning rate at first.
##' @param M a learning rate factor.
##' @param epoch_N the number of epoch to change learning rate.
##' @return lr_schedule
##' @author Satoshi Kume
##' @export lr_schedule
##' @examples
##' \dontrun{
##'
##' lr_reducer <- callback_learning_rate_scheduler(lr_schedule(lr_rate=0.01, epoch_N=10))
##'
##' }

lr_schedule <- function(epoch,
                        lr_rate=0.01,
                        epoch_N=10,
                        M=c(10, 100, 100)){
if(epoch <= epoch_N) {
    lr_rate
  } else if(epoch > epoch_N && epoch <= epoch_N*2){
    lr_rate/M[1]
  } else if(epoch > epoch_N*2 && epoch <= epoch_N*3){
    lr_rate/M[2]
  } else {
    lr_rate/M[3]
  }
}

