# 1.  Set up the model by specifying what type of model, setting its engine, and setting its mode (which was always classification)
# K NEAREST NEIGHBORS
# Tuning the number of neighbors
knn_mod <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")


# Logistic-Regression
logreg_mod <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")


# Elastic net log regression
# tuning penalty and mixture.
en_mod <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("classification") %>%
  set_engine("glmnet")


# RANDOM FOREST
# Tuning mtry (number of predictors), trees, and min_n (number of minimum values in each node)
rf_mod <- rand_forest(mtry = tune(), 
                      trees = tune(), 
                      min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")



# 2.  Set up the workflow for the model and add the model and the recipe.
knn_wf <- workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(car_recipe)

logreg_wf <- workflow() %>% 
  add_model(logreg_mod) %>% 
  add_recipe(car_recipe)

en_wf <- workflow() %>% 
  add_recipe(car_recipe) %>% 
  add_model(en_mod)
write_rds(en_wf, file = "/Users/wenxinjiang/Desktop/PSTAT 131/Wenxin 131project/tuned_models/en_wflow.rda")

rf_wf <- workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(car_recipe)



# 3.  Create a tuning grid to specify the ranges of the parameters you wish to tune as well as how many levels of each.
knn_grid <- grid_regular(neighbors(range = c(1,10)), levels = 10)
# No grid for logistic regression because no tuning parameters
en_grid <- grid_regular(penalty(), mixture(range = c(0,1)), levels = 10)
rf_grid <- grid_regular(mtry(range = c(1,8)), trees(range = c(200, 600)),
                        min_n(range = c(10, 20)),levels = 8)



# 4.  Tune the model and specify the workflow, k-fold cross validation folds, and the tuning grid for our chosen parameters to tune.
# K NEAREST NEIGHBORS
knn_tune <- tune_grid(
  knn_wf,
  resamples = car_folds,
  grid = knn_grid,
  control = control_grid(verbose = TRUE)
)


# Logistic-Regression
logreg_fit <- logreg_wf %>%
  fit_resamples(resamples = car_folds)

# ELASTIC NET
en_tune <- tune_grid(
  en_wf,
  resamples = car_folds,
  grid = en_grid,
  control = control_grid(verbose = TRUE)
)

# RANDOM FOREST
rf_tune <- tune_grid(
  rf_wf,
  resamples = car_folds,
  grid = rf_grid
)


# 5.  Save the tuned models to an RDS file to avoid rerunning the model.
# write_rds() to save

# logistic regression
write_rds(logreg_fit, file = "/Users/wenxinjiang/Desktop/PSTAT 131/Wenxin 131project/tuned_models/logreg.rda")

# K NEAREST NEIGHBORS
write_rds(knn_tune, file = "/Users/wenxinjiang/Desktop/PSTAT 131/Wenxin 131project/tuned_models/knn.rda")

# ELASTIC NET
write_rds(en_tune, file = "/Users/wenxinjiang/Desktop/PSTAT 131/Wenxin 131project/tuned_models/en.rda")

# RANDOM FOREST
write_rds(rf_tune, file = "/Users/wenxinjiang/Desktop/PSTAT 131/Wenxin 131project/tuned_models/rf.rda")



# 6.  Load back in the saved files.



# 7.  Collect the metrics of the tuned models, arrange in ascending order of mean to see what the highest ROC_AUC for that tuned model is, and slice to choose only the highest ROC_AUC. Save the ROC_AUC to a variable for comparison.
# collect_metrics() to collect the RUC_AUC
# slice() to save only RUC_AUC

# logistic REGRESSION 
logreg_auc <- collect_metrics(logreg_fited) %>% 
  arrange(desc(mean)) %>% 
  slice(1)

# K NEAREST NEIGHBORS
knn_auc <- collect_metrics(knn_tuned) %>% 
  arrange(desc(mean)) %>% 
  slice(1)

# ELASTIC NET
en_auc <- collect_metrics(en_tuned) %>% 
  arrange(desc(mean)) %>% 
  slice(1)

# RANDOM FOREST 
rf_auc <- collect_metrics(rf_tuned) %>% 
  arrange(desc(mean)) %>% 
  slice(1)

