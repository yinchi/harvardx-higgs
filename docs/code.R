## ----Load-libraries--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# R 4.1 key features: new pipe operator, \(x) as shortcut for function(x)
# R 4.0 key features: stringsAsFactors = FALSE by default, raw character strings r"()"
if (packageVersion('base') < '4.1.0') {
  stop('This code requires R >= 4.1.0!')
}

if(!require("pacman")) install.packages("pacman")
library(pacman)
p_load(data.table, dtplyr, tidyverse, R.utils, Rfast,
       lightgbm, keras, caret, pROC, knitr, conflicted)
conflict_prefer('summarize', 'dplyr')
conflict_prefer('summarise', 'dplyr')
conflict_prefer('filter', 'dplyr')
conflict_prefer('between', 'dplyr')
conflict_prefer('auc', 'pROC')

if(!is_keras_available()) install_keras()

# Somehow, this seems to prevent memory leaks
tensorflow::tf$compat$v1$disable_eager_execution()


## ----Download-data, results='hide'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
options(timeout=1800) # Give more time for the download to complete
if(!file.exists('data_raw/HIGGS.csv')) {
  download.file(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
    'data_raw/HIGGS.csv.gz')
  gunzip('data_raw/HIGGS.csv.gz')
}

## ----Load-Data-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load dataset into memory
higgs_all <- fread('data_raw/HIGGS.csv')

# Assign column names (csv contains no headers)
colnames(higgs_all) <- c('signal',
                     'lepton_pT', 'lepton_eta', 'lepton_phi',
                     'missing_E_mag', 'missing_E_phi',
                     'jet1_pT', 'jet1_eta', 'jet1_phi', 'jet1_btag',
                     'jet2_pT', 'jet2_eta', 'jet2_phi', 'jet2_btag',
                     'jet3_pT', 'jet3_eta', 'jet3_phi', 'jet3_btag',
                     'jet4_pT', 'jet4_eta', 'jet4_phi', 'jet4_btag',
                     'm_jj', 'm_jjj', 'm_lv', 'm_jlv',
                     'm_bb', 'm_wbb', 'm_wwbb')

# Separate input and output columns
xAll <- higgs_all %>% select(-signal) %>% as.data.table()
yAll <- higgs_all %>% select(signal) %>% as.data.table()
rm(higgs_all)


## ----Create-splits, results='hide'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create 10% test set split
set.seed(1)
idx <- createDataPartition(yAll$signal, p = 0.1, list = F)
gc()

x <- xAll[-idx,]
y <- yAll[-idx,]
xFinalTest <- x[idx,]
yFinalTest <- y[idx,]
rm(xAll, yAll)
gc()


## ----Hist-momentum-features, fig.height=3.5, fig.width=7-------------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(c(contains('_pT'), 'missing_E_mag')) |>
  as.data.frame() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free", nrow = 2) +
  ggtitle('Momentum features')


## ----Hist-momentum-features-log, fig.height=3.5, fig.width=7---------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(c(contains('_pT'), 'missing_E_mag')) |>
  as.data.frame() |>
  log() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free", nrow = 2) +
  ggtitle('Momentum features (log transform)')


## ----Skew-momentum-features------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tibble(
  Feature = x |>
    select(c(contains('_pT'), 'missing_E_mag')) |>
    as.data.table() |>
    colnames(),
  Skewness = x |>
    select(c(contains('_pT'), 'missing_E_mag')) |>
    as.data.table() |>
    as.matrix() |>
    colskewness(),
  'Skewness (log)' = x |>
    select(c(contains('_pT'), 'missing_E_mag')) |>
    as.data.table() |>
    log() |>
    as.matrix() |>
    colskewness()
) |>
  kable(align = 'lrr', booktabs = T, linesep = '')


## ----Hist-angular-features, fig.height=5, fig.width=6----------------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(c(contains('_eta'), contains('_phi'))) |>
  as.data.frame() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free_y", nrow = 3) +
  xlim(-pi,pi) +
  ggtitle('Angular features')


## ----Hist-btags, fig.height=3.5, fig.width=4-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(contains('_btag')) |>
  as.data.frame() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free", nrow = 2) +
  ggtitle('b-tag features')


## ----btag-means------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# b-tag means
tibble(
Mean = x |>
    select(contains('_btag')) |> 
    as.data.table() |> 
    as.matrix() |> 
    colMeans(),
'Standard Deviation' = x |>
    select(contains('_btag')) |> 
    as.data.table() |> 
    as.matrix() |> 
    colVars(std=T)
) |>
  rownames_to_column('Feature') |>
  kable(align = 'lrr', booktabs = T, linesep = '')


## ----Hist-high-level-features, fig.height=5, fig.width=7-------------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(contains('m_')) |>
  as.data.frame() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free", nrow = 3) +
  ggtitle('High-level features')


## ----Hist-high-level-features-log, fig.height=5, fig.width=7---------------------------------------------------------------------------------------------------------------------------------------------------------

x |>
  select(contains('m_')) |>
  as.data.frame() |>
  log() |>
  gather() |>
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~key, scales = "free", nrow = 3) +
  ggtitle('High-level features (log transform)')


## ----Skew-high-level-features----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tibble(
  Feature = x |>
    select(contains('m_')) |>
    as.data.table() |>
    colnames(),
  Skewness = x |>
    select(contains('m_')) |>
    as.data.table() |>
    as.matrix() |>
    colskewness(),
  'Skewness (log)' = x |>
    select(contains('m_')) |>
    as.data.table() |>
    log() |>
    as.matrix() |>
    colskewness()
) |>
  kable(align = 'lrr', booktabs = T, linesep = '')


## ----Log-transform, results='hide'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Apply log transform
x <- x |>
  mutate(across(
    c(contains('m_'), contains('_pT'), contains('_mag')),
    log)) |>
  as.data.table()

# Convert to matrices
x <- x %>% as.matrix()
gc()


## ----Mean-SD-scaling-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

m <- colMeans(x)
sd <- colVars(x, std = T) # std=T -> compute st. dev. instead of variance
x <- scale(x, center = m, scale = sd)


## ----Extract-signal--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

y <- y$signal


## ----lgb-drop-rate, results='hide'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Drop rates
dr <- c(0, 0.01, 0.02, seq(0.025, 0.07, 0.005), 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.5)

# Although we are not using tensorflow here, this is a convenience function
# to set multiple seeds across R and Python (i.e. reticulate)
tensorflow::set_random_seed(42, disable_gpu = F)

if (!file.exists('cache/gbm_results.RData')) {
  # Train
  gbm_scores <- sapply(dr, function(d) {
    gc()
    params <- list(
      num_threads = 32, # hardware-dependent
      boosting = "dart",
      metric = "auc",
      learning_rate = 1.0, # found to be a good value, too large leads to instability
      seed = 42, # note, any other seeds with default values take priority
      
      drop_rate = d
    )
    gbmodel <- lgb.cv(
      params, x, label = y,
      nrounds = 100, nfold = 3, obj = 'binary', verbose = 0
    )
    gbmodel$best_score
  })
  
  # Save
  save(gbm_scores, file = 'cache/gbm_results.RData')
}

load('cache/gbm_results.RData')
tensorflow::set_random_seed(42, disable_gpu = F)


## ----lightgbm-AUC-plot, fig.height=3, fig.width=4--------------------------------------------------------------------------------------------------------------------------------------------------------------------

tibble(dr,gbm_scores) |>
  ggplot(aes(dr,gbm_scores)) +
  geom_line() +
  geom_point() +
  xlab('Drop rate') +
  ylab('AUC') +
  theme_gray()


## ----lightgbm-best-AUC-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cat('The best AUC of', max(gbm_scores),
    'is achieved with a drop rate of', dr[which.max(gbm_scores)], '.\n')


## ----lightgbm-min-AUC-in-range---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
min(gbm_scores[between(dr,0.04,0.065)])


## ----lightgbm-best-model, results='hide'-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

tensorflow::set_random_seed(42, disable_gpu = F)
if (!file.exists('cache/gbm_results_2.RData')) {
  
  # Get the best drop rate from before
  best_dr <- dr[which.max(gbm_scores)]
  
  # Train the model again
  params <- list(
    num_threads = 32, # hardware-dependent
    boosting = "dart",
    metric = "auc",
    learning_rate = 1.0, # found to be a good value, too large leads to instability
    seed = 42, # note, any other seeds with default values take priority
    
    drop_rate = best_dr
  )
  gbmodel <- lgb.train(
    params, lgb.Dataset(x, label = y),
    nrounds = 100, obj = 'binary', verbose = 0
  )
  
  # Compute AUC
  auc_lgb_best <- auc(y,predict(gbmodel,x)) |> as.numeric()
  
  # Save AUC and model
  save(auc_lgb_best, gbmodel, file='cache/gbm_results_2.RData')
  cat(lgb.dump(gbmodel), file='cache/lgb_model.json') # (almost) human-readable version
}

load('cache/gbm_results_2.RData')
lgb.restore_handle(gbmodel) # if model is needed, we need to fully restore it
tensorflow::set_random_seed(42, disable_gpu = F)


## ----lightgbm-best-model-AUC-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

auc_lgb_best


## ----lgb-importance, fig.height=4, fig.width=4-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

par(cex=0.7)
gbmodel |> lgb.importance() |> lgb.plot.importance(top_n = 10)


## ----Keras-history---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_keras_history <- function(x, y, depth, breadth,
                        dropout = 0.5, learning_rate=0.0002, epochs = 50) {
  model <- keras_model_sequential()
  model |>
    layer_dense(breadth, 'relu', input_shape = ncol(x)) |>
    layer_dropout(rate = dropout)
  
  # subsequent hidden layers
  if (depth > 1) {
    for (layer in seq(2,depth)) {
      model |> layer_dense(breadth, 'relu') |> layer_dropout(rate = dropout)
    }
  }
  
  # output layer (logistic activation function for binary classification)
  model |> layer_dense(1, 'sigmoid')
  
  # compile model
  model |>
    keras::compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_adam(learning_rate = learning_rate),
      metrics = metric_auc()
    )
  
  # a larger batch size trains faster but uses more GPU memory
  history <- model |>
    fit(x, y, epochs = epochs, batch_size = 8192, validation_split = 0.2)
  
  rm(model)
  gc()
  k_clear_session()
  tensorflow::tf$compat$v1$reset_default_graph()
  
  history
}


## ----Keras-learning-rate-effect, results='hide'----------------------------------------------------------------------------------------------------------------------------------------------------------------------
tensorflow::set_random_seed(42, disable_gpu = F)
if (!file.exists('cache/nn_results.RDdata')) {
  history1 <- train_keras_history(x, y, 3, 256, learning_rate = 1e-3)
  history2 <- train_keras_history(x, y, 3, 256, learning_rate = 5e-4)
  history3 <- train_keras_history(x, y, 3, 256, learning_rate = 2e-4)
  save(history1, history2, history3, file = 'cache/nn_results.RDdata')
}
load('cache/nn_results.RDdata')
tensorflow::set_random_seed(42, disable_gpu = F)


## ----fig.height=3, fig.width=5---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tibble(
  epoch = seq(50),
  `1e-3` = history1$metrics$val_loss,
  `5e-4` = history2$metrics$val_loss,
  `2e-4` = history3$metrics$val_loss) |>
  pivot_longer(-epoch,'lr',values_to = 'loss') |>
  mutate(lr = as.numeric(lr)) |> 
  ggplot(aes(epoch,loss, color=as.factor(lr))) +
  geom_line() +
  xlab('Epochs') +
  ylab('Binary cross-entropy') +
  labs(color='Learning Rate')


## ----Keras-tuning-function-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Tuning values given defaults are not included in our parameter search for this
# project.
train_keras_auc <- function(x, y,
                        depth, breadth,
                        dropout = 0.5, learning_rate = 0.0002,
                        epochs = 50) {
  cat('DNN: ', depth, 'x', breadth, '\n')
  
  model <- keras_model_sequential()
  
  # By default, Keras applies Glorot uniform initialization for weights
  # and zero initialization for biases. Glorot uniform initialization
  # samples weights from Uniform(-sqrt(6/n),sqrt(6/n)) where n is the
  # sum of in and out nodes between two input/hidden/output layers.
  
  # first hidden layer
  model |>
    layer_dense(units = breadth,
                activation = 'relu',
                input_shape = ncol(x)) |>
    layer_dropout(rate = dropout)
  
  # subsequent hidden layers
  if (depth > 1) {
    for (layer in seq(2,depth)) {
      model |>
        layer_dense(units = breadth, activation = 'relu') |>
        layer_dropout(rate = dropout)
    }
  }
  
  # output layer (logistic activation function for binary classification)
  model |>
    layer_dense(units = 1, activation = 'sigmoid')
  
  # compile model
  model |>
    keras::compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_adam(learning_rate = learning_rate),
      metrics = metric_auc()
    )
  
  # a larger batch size trains faster but uses more GPU memory
  history <- model |>
    fit(x, y,
        epochs = epochs, batch_size = 8192,
        validation_split = 0.2)
  
  ypred <- model |> predict(x, batch_size = 8192) |> as.vector()
  auc <- roc(y,ypred) |> auc() |> as.numeric()
  
  rm(model)
  gc()
  k_clear_session()
  tensorflow::tf$compat$v1$reset_default_graph()
  
  auc
}


## ----Keras-tuning-run, results='hide'--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Try NN training for different NN depths and breadths.  Cache using .RData file.
tensorflow::set_random_seed(42, disable_gpu = F)
if (!file.exists('cache/nn_results2.RDdata')) {
  nn_results <- tibble(depth = integer(), breadth = integer(), auc = numeric())
  
  for(l in 1:5) { # depth: number of hidden layers
    for (n in 2^c(5:11)) { # breadth: hidden nodes per layer
      nn_results <- nn_results |>
        add_row(depth = l,
                breadth = n,
                auc = train_keras_auc(x, y, l, n))
    }
  }
  save(nn_results, file = 'cache/nn_results2.RDdata')
}
load('cache/nn_results2.RDdata')
tensorflow::set_random_seed(42, disable_gpu = F)


## ----Keras-tuning-plot, fig.height=3.5, fig.width=4.5----------------------------------------------------------------------------------------------------------------------------------------------------------------

# heatmap of AUC vs depth and breadth
nn_results |> ggplot(aes(as.factor(depth), as.factor(breadth), fill = auc)) +
  geom_tile() +
  geom_text(aes(label = round(auc,4)), color = "black", size = 3) +
  scale_fill_viridis_c() +
  xlab('Depth (number of hidden layers)') +
  ylab('Breadth (hidden nodes per layer)')


## ----train-keras, eval=FALSE, include=FALSE--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 
## train_keras <- function(x, y,
##                         depth, breadth,
##                         dropout = 0.5, learning_rate = 0.0002,
##                         epochs = 50) {
## 
##   model <- keras_model_sequential()
## 
##   # first hidden layer
##   model |>
##     layer_dense(units = breadth,
##                 activation = 'relu',
##                 input_shape = ncol(x)) |>
##     layer_dropout(rate = dropout)
## 
##   # subsequent hidden layers
##   if (depth > 1) {
##     for (layer in seq(2,depth)) {
##       model |>
##         layer_dense(units = breadth, activation = 'relu') |>
##         layer_dropout(rate = dropout)
##     }
##   }
## 
##   # output layer (logistic activation function for binary classification)
##   model |>
##     layer_dense(units = 1, activation = 'sigmoid')
## 
##   # compile model
##   model |>
##     keras::compile(
##       loss = 'binary_crossentropy',
##       optimizer = optimizer_adam(learning_rate = learning_rate),
##       metrics = metric_auc()
##     )
## 
##   # a larger batch size trains faster but uses more GPU memory
##   model |>
##     fit(x, y,
##         epochs = epochs, batch_size = 8192,
##         validation_split = 0.2)
## }
## 
## tensorflow::set_random_seed(42, disable_gpu = F)
## if (!file.exists('cache/nn_results3.RDdata')) {
##   model_full <- train_keras(x, y, 3, 2048, learning_rate = 2e-4)
##   model_low_only <- train_keras(x[,1:21], y, 3, 2048, learning_rate = 2e-4)
##   save(history4, history5, file='cache/nn_results3.RDdata')
## }
## load('cache/nn_results3.RDdata')
## tensorflow::set_random_seed(42, disable_gpu = F)

