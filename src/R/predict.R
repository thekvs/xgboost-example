library(xgboost)

args <- commandArgs(TRUE)

model_file <- args[1]
data_file <- args[2]
result_file <- args[3]

data <- read.csv(data_file)
regressor <- xgb.load(model_file)

predictions <- predict(regressor, xgb.DMatrix(as.matrix(data)))
write.table(round(predictions, digits=3), file=result_file, sep=",",
            col.names=F, row.names=F)

