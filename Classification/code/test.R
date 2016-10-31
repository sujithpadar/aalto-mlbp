basedir <- "/media/Documents/01 Aalto/03 Study/Semester 01/02 Machine Learning Basic Principles - Alex Jung/Term Project/Classification/"
setwd(basedir)

data <- read.csv("data/classification_dataset_training.csv")
formula <- as.formula(paste("rating~",paste(setdiff(colnames(data),c("ID","rating")),collapse = "+"),sep=""))

mod <- glm(data=data,formula=formula,family = binomial("logit"))

predict <- predict(mod,type="response")
