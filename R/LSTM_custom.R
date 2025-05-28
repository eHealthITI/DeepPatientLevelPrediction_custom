# creates a list of model hyperparameters and returns a list with LSTM model settings.
setLstm <- function(numLayers = 1,
                    hiddenSize = 128,
                    dropout = 0.2,
                    bidirectional = TRUE,
                    estimatorSettings = setEstimator(),
                    hyperParamSearch = "random",
                    randomSample = 1,
                    randomSampleSeed = NULL) {

  # Input checks (like the ones in setTransformer)
  
  paramGrid <- list(
    numLayers = numLayers,
    hiddenSize = as.integer(hiddenSize),
    dropout = dropout,
    bidirectional = bidirectional
  )
  paramGrid <- c(paramGrid, estimatorSettings$paramsToTune)

  param <- PatientLevelPrediction::listCartesian(paramGrid)

  if (hyperParamSearch == "random") {
    suppressWarnings(withr::with_seed(randomSampleSeed, {
      param <- param[sample(length(param), randomSample)]
    }))
  }

  results <- list(
    fitFunction = "DeepPatientLevelPrediction::fitEstimator",
    param = param,
    estimatorSettings = estimatorSettings,
    saveType = "file",
    modelParamNames = c("numLayers", "hiddenSize", "dropout", "bidirectional"),
    modelType = "LSTM_custom"
  )
  attr(results$param, "settings")$modelType <- results$modelType
  class(results) <- "modelSettings"
  return(results)
}


# Provides a quick preset to create an LSTM model configuration using default hyperparameters
setDefaultLstm <- function(estimatorSettings = setEstimator(
                                      learningRate = "auto",
                                      weightDecay = 1e-4,
                                      batchSize = 128,
                                      epochs = 15,
                                      seed = NULL,
                                      device = "cpu"
                                    )) {
  lstmSettings <- setLstm(
    numLayers = 3,
    hiddenSize = as.integer(128),
    dropout = 0.2,
    bidirectional = TRUE,
    estimatorSettings = estimatorSettings,
    hyperParamSearch = "random",
    randomSample = 1
  )
  attr(lstmSettings, "settings")$name <- "LSTM_custom"
  return(lstmSettings)
}

