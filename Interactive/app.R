library(shiny)
library(shinythemes)
library(caret)
library(randomForest)
library(C50)
library(nnet)
library(kernlab)

# 1. LOAD ALL MODELS
# Ensure these are in the same folder as this script!
logreg_model <- readRDS("glm_model.rds")
knn_model    <- readRDS("knn_model_object.rds")
ann_model    <- readRDS("ann.rds")
svm_model    <- readRDS("svm_rbf.rds")
tree_model   <- readRDS("c50_model.rds")
meta_model   <- readRDS("depression_meta_model_rf.rds")

# 2. UI - User Inputs
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("TO414: Depression Prediction Portal"),
  
  sidebarLayout(
    sidebarPanel(
      h4("User Profile"),
      numericInput("age_raw", "Age:", value = 21, min = 18, max = 90),
      
      selectInput("marital", "Marital Status:", 
                  choices = c("Single", "Married", "Divorced", "Widowed")),
      
      selectInput("edu", "Education Level:", 
                  choices = c("High School", "Bachelors", "Masters", "Ph.D", "Associate Degree")),
      
      numericInput("income_raw", "Annual Income ($):", value = 50000),
      numericInput("children_raw", "Number of Children:", value = 0, min = 0),
      
      selectInput("smoke", "Smoking Status:", choices = c("Non-smoker", "Former", "Current")),
      selectInput("activity", "Physical Activity:", choices = c("Sedentary", "Moderate", "Active")),
      selectInput("employ", "Employment Status:", choices = c("Employed", "Unemployed")),
      selectInput("alcohol", "Alcohol Consumption:", choices = c("Low", "Moderate", "High")),
      selectInput("diet", "Dietary Habits:", choices = c("Healthy", "Moderate", "Unhealthy")),
      selectInput("sleep", "Sleep Quality:", choices = c("Good", "Fair", "Poor")),
      
      hr(),
      h4("History"),
      checkboxInput("substance", "History of Substance Abuse", value = FALSE),
      checkboxInput("fam_dep", "Family History of Depression", value = FALSE),
      checkboxInput("chronic", "Chronic Medical Conditions", value = FALSE),
      
      actionButton("predict_btn", "Consult the Council", class = "btn-primary btn-lg", style="width: 100%;")
    ),
    
    mainPanel(
      wellPanel(
        h3("Final Meta-Model Verdict:", align = "center"),
        h1(textOutput("final_result"), align = "center")
      ),
      hr(),
      h4("Expert Probability Breakdown"),
      tableOutput("expert_probs"),
      p(em("This model aggregates 5 distinct machine learning 'experts' to reach a consensus."))
    )
  )
)

# 3. SERVER - Logic
server <- function(input, output) {
  
  # Replace 100 and 200000 with the actual MAX values from your training data!
  scale_val <- function(val, max_val) { val / max_val }
  
  prediction_data <- eventReactive(input$predict_btn, {
    
    # BUILD THE SUPER-DATAFRAME
    # This includes BOTH the 'dummy' names and the 'original' names
    new_user <- data.frame(
      # --- Original Names (for Models that want factors/dots) ---
      Age = as.integer(input$age_raw),
      Marital.Status = factor(input$marital, levels = c("Divorced", "Married", "Single", "Widowed")),
      Education.Level = factor(input$edu, levels = c("Associate Degree", "Bachelors", "High School", "Masters", "Ph.D")),
      Number.of.Children = as.integer(input$children_raw),
      Smoking.Status = factor(input$smoke, levels = c("Current", "Former", "Non-smoker")),
      Physical.Activity.Level = factor(input$activity, levels = c("Active", "Moderate", "Sedentary")),
      Employment.Status = factor(input$employ, levels = c("Employed", "Unemployed")),
      Income = as.numeric(input$income_raw),
      Alcohol.Consumption = factor(input$alcohol, levels = c("High", "Low", "Moderate")),
      Dietary.Habits = factor(input$diet, levels = c("Healthy", "Moderate", "Unhealthy")),
      Sleep.Patterns = factor(input$sleep, levels = c("Fair", "Good", "Poor")),
      History.of.Substance.Abuse = factor(if(input$substance) "Yes" else "No", levels = c("No", "Yes")),
      Family.History.of.Depression = factor(if(input$fam_dep) "Yes" else "No", levels = c("No", "Yes")),
      Chronic.Medical.Conditions = factor(if(input$chronic) "Yes" else "No", levels = c("No", "Yes")),
      
      # --- Dummy Names (for Models that want scaled/underscores) ---
      age = scale_val(input$age_raw, 90),
      marital_status_divorced = if(input$marital == "Divorced") 1 else 0,
      marital_status_married = if(input$marital == "Married") 1 else 0,
      marital_status_single = if(input$marital == "Single") 1 else 0,
      marital_status_widowed = if(input$marital == "Widowed") 1 else 0,
      education_level_bachelors_degree = if(input$edu == "Bachelors") 1 else 0,
      education_level_high_school = if(input$edu == "High School") 1 else 0,
      education_level_masters_degree = if(input$edu == "Masters") 1 else 0,
      education_level_ph_d = if(input$edu == "Ph.D") 1 else 0,
      number_of_children = scale_val(input$children_raw, 10),
      smoking_status_former = if(input$smoke == "Former") 1 else 0,
      smoking_status_non_smoker = if(input$smoke == "Non-smoker") 1 else 0,
      physical_activity_level_moderate = if(input$activity == "Moderate") 1 else 0,
      physical_activity_level_sedentary = if(input$activity == "Sedentary") 1 else 0,
      employment_status_unemployed = if(input$employ == "Unemployed") 1 else 0,
      income = scale_val(input$income_raw, 200000),
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
    
    # 1. Get Base Probabilities
    # Note: Added 'tryCatch' so one bad model doesn't crash the whole app
    p_logreg <- tryCatch(predict(logreg_model, new_user, type = "response"), error = function(e) 0)
    p_knn    <- tryCatch(predict(knn_model,    new_user, type = "prob")[,2], error = function(e) 0)
    p_ann    <- tryCatch(predict(ann_model,    new_user, type = "raw")[,1], error = function(e) 0)
    p_svm    <- tryCatch(predict(svm_model,    new_user, type = "probabilities")[,2], error = function(e) 0)
    p_tree   <- tryCatch(predict(tree_model,   new_user, type = "prob")[,2], error = function(e) 0)
    
    # 2. Level 1 Stack
    live_stack <- data.frame(
      logreg = as.numeric(p_logreg),
      knn    = as.numeric(p_knn),
      ann    = as.numeric(p_ann),
      svm    = as.numeric(p_svm),
      tree   = as.numeric(p_tree)
    )
    
    # 3. Meta-Prediction
    final_pred <- predict(meta_model, live_stack)
    
    list(final = final_pred, probs = live_stack)
  })
  
  output$final_result <- renderText({
    res <- prediction_data()$final
    if(res == "1" || res == "Yes") "PREDICTED: High Risk" else "PREDICTED: Low Risk"
  })
  
  output$expert_probs <- renderTable({
    prediction_data()$probs
  })
}

shinyApp(ui = ui, server = server)