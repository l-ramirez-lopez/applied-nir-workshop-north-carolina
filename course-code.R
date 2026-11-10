library(proximetricsR)

mlocal_cal <- proxiscout_read_data(
  "data/local-samples-spain/local-spectra-cal.xlsx", 
  references_file = "data/local-samples-spain/local-properties-cal.xlsx"
)

mlocal_val <- proxiscout_read_data(
  "data/local-samples-spain/local-spectra-val.xlsx", 
  references_file = "data/local-samples-spain/local-properties-val.xlsx"
)

mlibrary <- proxiscout_read_data(
  "data/spectral-library-usa/spectral-library-spectra.xlsx",
  references_file = "data/spectral-library-usa/spectral-library-properties.xlsx"
)

dim(mlocal_cal)
colnames(mlocal_cal)
View(mlocal_cal)
View(mlocal_cal$spc)

wavs <- as.numeric(colnames(mlocal_cal$spc))
wavs
matplot(
  wavs,
  t(mlocal_cal$spc), 
  type = "l", 
  lty = 1,
  col = rgb(1, 0, 0, 0.5),
  xlab = expression("Wavenumber ("*cm^{-1}*")"),
  ylab = "Reflectance",
  main = "Calibration data",
  xlim = c(7500, 4000)
)

my_formulas <- list(OC ~ spc)
my_formulas_all <- list(OC ~ spc, OM ~ spc)
prep_snv()
matplot(
  wavs,
  t(mlocal_val$spc), 
  type = "l", 
  lty = 1,
  col = rgb(1, 0, 0, 0.5),
  xlab = expression("Wavenumber ("*cm^{-1}*")"),
  ylab = "Reflectance",
  main = "Validation data",
  xlim = c(7500, 4000)
)




recipe_1 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_snv(),
  prep_derivative(m = 1, w = 11, p = 1, algorithm = "savitzky-golay"),
  device = "proxiscout" # !!!
)
recipe_1

recipe_2 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_snv(),
  prep_derivative(m = 2, w = 11, p = 2, algorithm = "savitzky-golay"),
  device = "proxiscout" 
)

recipe_3 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_detrend(p = 2),
  prep_derivative(m = 2, w = 11, p = 2, algorithm = "savitzky-golay"),
  device = "proxiscout" 
)

recipe_4 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_transform(to = "absorbance"),
  prep_derivative(m = 2, w = 11, p = 2, algorithm = "savitzky-golay"),
  device = "proxiscout" 
)

recipe_5 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_transform(to = "absorbance"),
  prep_derivative(m = 1, w = 11, p = 2, algorithm = "savitzky-golay"),
  prep_snv(),
  device = "proxiscout" 
)

recipe_6 <- preprocess_recipe(
  prep_resample(grid = "proxiscout"),
  prep_transform(to = "absorbance"),
  prep_derivative(m = 2, w = 11, p = 2, algorithm = "savitzky-golay"),
  prep_snv(),
  device = "proxiscout" 
)

my_preprocessings <- list(recipe_1, recipe_2, recipe_3, recipe_4, recipe_5)

fit_1 <- fit_plsr(ncomp = 10, type = "standard")
fit_2 <- fit_plsr(ncomp = 9, type = "modified") 

my_fittings <- list(fit_1, fit_2)


my_control <- calibration_control(
  validation_type = "kfold", # kfold cross validation 
  number = 5, # 5 folds
  folds = "random", # folds made at random
  remove_outliers = 1 # fit, remove outliers and fit again (one refit iteration)
)

my_model <- calibrate_models(
  formulas = my_formulas,
  data = mlocal_cal,
  preprocess_recipes = my_preprocessings,
  methods = my_fittings,
  control = my_control
)


help(plot.spectral_model)
plot(
  my_model$final_models$`OC ~ spc`,  
  spectral = "all", 
  cv = "all", 
  validation = "all"
)
one_model <- my_model$final_models$`OC ~ spc`
y_hat <- predict(one_model, newdata = mlocal_val)

axes_range <- range(c(y_hat$predictions, mlocal_val$OC), na.rm = TRUE)

plot(
  y_hat$predictions, 
  mlocal_val$OC, 
  xlab = "Predicted OC (%)", 
  ylab = "Reference OC (%)", 
  main = "External validation",
  xlim = axes_range,
  ylim = axes_range, 
  pch = 16, 
  cex = 2, 
  col = "#F0AD00"
)
abline(0, 1, col = "red")
val_results <- validate_prediction(
  prediction = y_hat, 
  reference = mlocal_val$OC
)
val_results$validation$ncomp_4$val_stats
dir.create("outputs", showWarnings = FALSE)

proxiscout_write_model(
  one_model, 
  file = "outputs/local_oc_model.json"
)
