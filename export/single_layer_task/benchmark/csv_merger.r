csv.merger = function(files.list, models.list) {
  # read in each file, rename the model to correct name, and concat all the tables row-wise
  merge.csvs = function(load.files.names, model.names) {
    combined.tables <- NULL
    # load tables for each element in the list AND EXPAND THEM
    tables <- lapply(lapply(load.files.names, read_csv), expand.name)
    for (idx in 1:length(tables)) {
      t <- ldply(tables[idx], data.frame)  # convert from list to df
      # create model col only if a list of model names has been given. Otherwise will use predefined names from _single_layer_task_expand_name.r
      if (length(model.names) != 0) {
        t$model <- model.names[[idx]]      # rename the model name to pre-defined value in list
      }
      combined.tables <- rbind(combined.tables, t)  # add model data to an accumulated table
    }
    return(combined.tables)
  }

  csvs.combined = merge.csvs(files.list, models.list)
  # dat needs model col to be a factor (because different models = different levels).
  # Without this line, you can't drop levels when plotting
  csvs.combined$model <- as.factor(as.vector(csvs.combined$model))
  return(csvs.combined)
}

load.and.merge.csvs = function(lookup.name) {
  return(switch(
    lookup.name,
    "None" = expand.name(read_csv(name.file)),
    "benchmark_sltr_mul" = csv.merger(list(
      paste0(load_folder, 'MNAC', csv_ext),
      paste0(load_folder, 'NALU', csv_ext),
      paste0(load_folder, 'NMU', csv_ext),
      paste0(load_folder, 'NPU', csv_ext),
      paste0(load_folder, 'RealNPU', csv_ext),
      paste0(load_folder, 'iNALU', csv_ext),
      paste0(load_folder, 'G-NALU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "benchmark_sltr_div" = csv.merger(list(
      paste0(load_folder, 'MNAC', csv_ext),
      paste0(load_folder, 'NALU', csv_ext),
      paste0(load_folder, 'NPU', csv_ext),
      paste0(load_folder, 'RealNPU', csv_ext),
      paste0(load_folder, 'iNALU', csv_ext),
      paste0(load_folder, 'G-NALU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "benchmark_sltr_add" = csv.merger(list(
      paste0(load_folder, 'NAC', csv_ext),
      paste0(load_folder, 'NALU', csv_ext),
      paste0(load_folder, 'NAU', csv_ext),
      paste0(load_folder, 'iNALU', csv_ext),
      paste0(load_folder, 'G-NALU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "benchmark_sltr_sub" = csv.merger(list(
      paste0(load_folder, 'NAC', csv_ext),
      paste0(load_folder, 'NALU', csv_ext),
      paste0(load_folder, 'NAU', csv_ext),
      paste0(load_folder, 'iNALU', csv_ext),
      paste0(load_folder, 'G-NALU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "nalu" = csv.merger(list(
      paste0(load_folder, 'NALU', csv_ext),
      paste0(load_folder, 'iNALU', csv_ext),
      paste0(load_folder, 'G-NALU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "nau-failure" = csv.merger(list(
      paste0(load_folder, '/add/', 'NAU', csv_ext),
      paste0(load_folder, '/sub/', 'NAU', csv_ext)
    ),
      list('NAU (add)', 'NAU (sub)')
    ),
    "benchmark_sltr_mul" = csv.merger(list(
      paste0(load_folder, 'NMU', csv_ext),
      paste0(load_folder, 'MCFC', csv_ext),
      paste0(load_folder, 'MulMCFCSignINALU', csv_ext),
      paste0(load_folder, 'MulMCFCSignRealNPU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "benchmark_sltr_mul_mcfc" = csv.merger(list(
      paste0(load_folder, 'NMU', csv_ext),
      paste0(load_folder, 'MulMCFC', csv_ext),
      paste0(load_folder, 'MulMCFCSignINALU', csv_ext),
      paste0(load_folder, 'MulMCFCSignRealNPU', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    "benchmark_sltr_add_mcfc" = csv.merger(list(
      paste0(load_folder, 'NAU', csv_ext),
      paste0(load_folder, 'MCFC', csv_ext)
    ),
      list()  # use predefined names from _single_layer_task_expand_name.r (allows for latex notation)
    ),
    stop("Key given to csv_merger does not exist!")
  ))
}








