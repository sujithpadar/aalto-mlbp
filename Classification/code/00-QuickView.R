basedir <- "/mnt/B4D486CBD4868EF4/01. Aalto/03. Study/Semester 01/02. Machine Learning Basic Principles - Alex Jung/Exercises/Term Project/Classification/data"
setwd(basedir)

df <- read.csv("classification_dataset_training.csv")

source("/mnt/B4D486CBD4868EF4/01. Aalto/06 R/00 Macros/macros.R")
source("/mnt/B4D486CBD4868EF4/01. Aalto/06 R/00 Macros/plotMacros.R")


colavars <- colnames(df)[-1]

sapply(colavars,function(x){
  plotAVE(df,x,act="rating",subdir = "charts")
})

plotAVEsplit(df,var="good",byvar = "but",act="rating",subdir = "charts")
