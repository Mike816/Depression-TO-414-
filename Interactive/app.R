library(shiny)
library(shinythemes)
library(caret)
library(randomForest)
library(C50)
library(neuralnet)
library(kernlab)

# ---------------------------------------------------------
# 1. LOAD ALL MODELS & REFERENCE DATA
# ---------------------------------------------------------
logreg_model <- readRDS("glm_model.rds")
knn_model    <- readRDS("knn_model_object.rds")
ann_model    <- readRDS("ann.rds")
svm_model    <- readRDS("svm_rbf.rds")
tree_model   <- readRDS("c50_model.rds")
rf_model     <- readRDS("rf_model.rds")

# CRITICAL: Ensure this meta-model file actually exists and matches your 6 experts!
# If you haven't re-trained it since adding RF, this will crash.
meta_model   <- readRDS("depression_meta_model_tree.rds") 

train_og <- readRDS("train_original.rds")

# Boundaries
age_min <- min(train_og$Age); age_max <- max(train_og$Age)
inc_min <- min(train_og$Income); inc_max <- max(train_og$Income)
chi_min <- min(train_og$Number.of.Children); chi_max <- max(train_og$Number.of.Children)

# ---------------------------------------------------------
# 2. UI
# ---------------------------------------------------------
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("TO414: Depression Prediction Portal"),
  sidebarLayout(
    sidebarPanel(
      h4("User Profile"),
      numericInput("age_raw", "Age:", value = 21, min = 18, max = 90),
      selectInput("marital", "Marital Status:", choices = levels(train_og$Marital.Status)),
      selectInput("edu", "Education Level:", choices = levels(train_og$Education.Level)),
      numericInput("income_raw", "Annual Income ($):", value = 50000),
      numericInput("children_raw", "Number of Children:", value = 0, min = 0),
      selectInput("smoke", "Smoking Status:", choices = levels(train_og$Smoking.Status)),
      selectInput("activity", "Physical Activity:", choices = levels(train_og$Physical.Activity.Level)),
      selectInput("employ", "Employment Status:", choices = levels(train_og$Employment.Status)),
      selectInput("alcohol", "Alcohol Consumption:", choices = levels(train_og$Alcohol.Consumption)),
      selectInput("diet", "Dietary Habits:", choices = levels(train_og$Dietary.Habits)),
      selectInput("sleep", "Sleep Quality:", choices = levels(train_og$Sleep.Patterns)),
      hr(),
      h4("History"),
      checkboxInput("substance", "History of Substance Abuse", value = FALSE),
      checkboxInput("fam_dep", "Family History of Depression", value = FALSE),
      checkboxInput("chronic", "Chronic Medical Conditions", value = FALSE),
      actionButton("predict_btn", "Consult the Council", class = "btn-primary btn-lg", style="width: 100%;")
    ),
    mainPanel(
      wellPanel(
        h3("Final Stacked Model Prediction:", align = "center"),
        h1(textOutput("final_result"), align = "center")
      ),
      hr(),
      h4("Individual Model Probabilities"),
      tableOutput("expert_probs"),
      p(em("This stacked model combines predictions from 6 distinct models and uses a decision tree as a second layer"))
    )
  )
)

# ---------------------------------------------------------
# 3. SERVER
# ---------------------------------------------------------
server <- function(input, output) {
  
  calc_minmax <- function(val, min_val, max_val) { (val - min_val) / (max_val - min_val) }
  
  prediction_data <- eventReactive(input$predict_btn, {
    
    # 1. THE DATA FRAME
    new_user <- data.frame(
      # --- Original Names ---
      Age = as.integer(input$age_raw),
      Marital.Status = factor(input$marital, levels = levels(train_og$Marital.Status)),
      Education.Level = factor(input$edu, levels = levels(train_og$Education.Level)),
      Number.of.Children = as.integer(input$children_raw),
      Smoking.Status = factor(input$smoke, levels = levels(train_og$Smoking.Status)),
      Physical.Activity.Level = factor(input$activity, levels = levels(train_og$Physical.Activity.Level)),
      Employment.Status = factor(input$employ, levels = levels(train_og$Employment.Status)),
      Income = as.numeric(input$income_raw),
      Alcohol.Consumption = factor(input$alcohol, levels = levels(train_og$Alcohol.Consumption)),
      Dietary.Habits = factor(input$diet, levels = levels(train_og$Dietary.Habits)),
      Sleep.Patterns = factor(input$sleep, levels = levels(train_og$Sleep.Patterns)),
      History.of.Substance.Abuse = factor(if(input$substance) "Yes" else "No", levels = levels(train_og$History.of.Substance.Abuse)),
      Family.History.of.Depression = factor(if(input$fam_dep) "Yes" else "No", levels = levels(train_og$Family.History.of.Depression)),
      Chronic.Medical.Conditions = factor(if(input$chronic) "Yes" else "No", levels = levels(train_og$Chronic.Medical.Conditions)),
      
      # --- Dummy Names ---
      age = calc_minmax(input$age_raw, age_min, age_max),
      marital_status_divorced = if(input$marital == "Divorced") 1 else 0,
      marital_status_married = if(input$marital == "Married") 1 else 0,
      marital_status_single = if(input$marital == "Single") 1 else 0,
      marital_status_widowed = if(input$marital == "Widowed") 1 else 0,
      education_level_bachelors_degree = if(input$edu == "Bachelors") 1 else 0,
      education_level_high_school = if(input$edu == "High School") 1 else 0,
      education_level_masters_degree = if(input$edu == "Masters") 1 else 0,
      education_level_ph_d = if(input$edu == "Ph.D") 1 else 0,
      number_of_children = calc_minmax(input$children_raw, chi_min, chi_max),
      smoking_status_former = if(input$smoke == "Former") 1 else 0,
      smoking_status_non_smoker = if(input$smoke == "Non-smoker") 1 else 0,
      physical_activity_level_moderate = if(input$activity == "Moderate") 1 else 0,
      physical_activity_level_sedentary = if(input$activity == "Sedentary") 1 else 0,
      employment_status_unemployed = if(input$employ == "Unemployed") 1 else 0,
      income = calc_minmax(input$income_raw, inc_min, inc_max),
      alcohol_consumption_low = if(input$alcohol == "Low") 1 else 0,
      alcohol_consumption_moderate = if(input$alcohol == "Moderate") 1 else 0,
      dietary_habits_moderate = if(input$diet == "Moderate") 1 else 0,
      dietary_habits_unhealthy = if(input$diet == "Unhealthy") 1 else 0,
      sleep_patterns_good = if(input$sleep == "Good") 1 else 0,
      sleep_patterns_poor = if(input$sleep == "Poor") 1 else 0,
      history_of_substance_abuse_yes = if(input$substance) 1 else 0,
      family_history_of_depression_yes = if(input$fam_dep) 1 else 0,
      chronic_medical_conditions_yes = if(input$chronic) 1 else 0
    )
    
    # 2. FILTER COLUMN SETS
    dummy_list <- colnames(knn_model$learn$X)
    dummy_input <- new_user[, dummy_list]
    
    # Use the rownames of importance for RF/Tree/GLM features
    rf_list <- rownames(rf_model$importance)
    rf_input <- new_user[, rf_list]
    
    # 3. PREDICTIONS
    p_logreg <- tryCatch(predict(logreg_model, rf_input, type = "response"), error = function(e) 0)
    p_knn    <- tryCatch(predict(knn_model,    dummy_input, type = "prob")[,2], error = function(e) 0)
    
    # Change this block in your server code:
    p_ann <- tryCatch({
      # neuralnet objects use predict() without a 'type' argument
      # It returns a matrix, so we take the first column
      res <- predict(ann_model, dummy_input) 
      as.numeric(res[, 1]) 
    }, error = function(e) {
      # Log the error so you can see it in the browser logs
      message("ANN Prediction Error: ", e$message)
      return(0)
    })
    
    p_svm    <- tryCatch(predict(svm_model,    as.matrix(dummy_input), type = "probabilities")[,2], error = function(e) 0)
    p_tree   <- tryCatch(predict(tree_model,   rf_input, type = "prob")[,2], error = function(e) 0)
    p_rf     <- tryCatch(predict(rf_model,     rf_input, type = "prob")[, "Yes"], error = function(e) 0)
    
    # 4. THE STACK
    live_stack <- data.frame(
      logreg = as.numeric(p_logreg),
      knn    = as.numeric(p_knn),
      ann    = as.numeric(p_ann),
      svm    = as.numeric(p_svm),
      tree   = as.numeric(p_tree),
      rf     = as.numeric(p_rf)
    )
    
    # 5. META-MODEL
    # This will fail if meta_model was trained on 5 columns but sees 6!
    final_pred <- tryCatch(predict(meta_model, live_stack), error = function(e) "Error")
    
    list(final = final_pred, probs = live_stack)
  })
  
  output$final_result <- renderText({
    res <- prediction_data()$final
    if(res == "Error") return("Error: Meta-Model Mismatch (Check Expert Count)")
    if(res == "1" || res == "Yes") "PREDICTED: Depressed" else "PREDICTED: Not Depressed"
  })
  
  output$expert_probs <- renderTable({ prediction_data()$probs })
}

shinyApp(ui = ui, server = server)