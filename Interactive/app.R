library(shiny)
library(shinythemes)
library(caret)
library(randomForest)
library(C50)
library(nnet)
library(kernlab)

# 1. LOAD MODELS
# Make sure these filenames match exactly what you saved
logreg_model <- readRDS("glm_model.rds")
knn_model    <- readRDS("knn_model_object.rds")
ann_model    <- readRDS("ann.rds")
svm_model    <- readRDS("svm_rbf.rds")
tree_model   <- readRDS("c50_model.rds")
meta_model   <- readRDS("depression_meta_model_rf.rds")

# 2. UI
ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("Depression Prediction Portal: Council of Experts"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Demographics & Lifestyle"),
      numericInput("age_raw", "Age:", value = 21, min = 18, max = 80),
      selectInput("marital", "Marital Status:", 
                  choices = c("Single", "Married", "Divorced", "Widowed")),
      selectInput("edu", "Education Level:", 
                  choices = c("High School", "Bachelors", "Masters", "Ph.D")),
      numericInput("income_raw", "Annual Income ($):", value = 50000),
      numericInput("children_raw", "Number of Children:", value = 0, min = 0),
      
      hr(),
      h4("Health Factors"),
      selectInput("smoke", "Smoking Status:", choices = c("Non-smoker", "Former", "Current")),
      selectInput("activity", "Physical Activity:", choices = c("Sedentary", "Moderate", "Active")),
      selectInput("diet", "Dietary Habits:", choices = c("Healthy", "Moderate", "Unhealthy")),
      selectInput("sleep", "Sleep Quality:", choices = c("Good", "Moderate", "Poor")),
      selectInput("alcohol", "Alcohol Consumption:", choices = c("None", "Low", "Moderate", "High")),
      
      hr(),
      h4("Medical History"),
      checkboxInput("substance", "History of Substance Abuse", value = FALSE),
      checkboxInput("fam_dep", "Family History of Depression", value = FALSE),
      checkboxInput("chronic", "Chronic Medical Conditions", value = FALSE),
      
      actionButton("predict_btn", "Run Consensus Prediction", class = "btn-primary btn-lg", style="width: 100%;")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Result", 
                 br(),
                 wellPanel(
                   h3("Meta-Model Verdict:", align = "center"),
                   h1(textOutput("final_result"), align = "center")
                 ),
                 hr(),
                 h4("Expert Probability Breakdown"),
                 tableOutput("expert_probs")
        ),
        tabPanel("Technical Info",
                 p("This model uses a stacked ensemble (Random Forest) to combine predictions from:"),
                 tags$ul(
                   tags$li("Logistic Regression"),
                   tags$li("K-Nearest Neighbors (k=15)"),
                   tags$li("Artificial Neural Network"),
                   tags$li("Support Vector Machine (RBF)"),
                   tags$li("C5.0 Decision Tree")
                 ))
      )
    )
  )
)

# 3. SERVER
server <- function(input, output) {
  
  # Helper function to mimic your training scaling
  # NOTE: You should replace these denominators with the max values from your original dataset
  scale_val <- function(val, max_val) { val / max_val }
  
  prediction_data <- eventReactive(input$predict_btn, {
    
    # Create the single-row dataframe matching your train_dummy structure
    # We set everything to 0 first, then toggle the "1"s based on input
    new_user <- data.frame(
      age = scale_val(input$age_raw, 80), # Scaling based on your num spread
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
      employment_status_unemployed = 0, # Placeholder, add input if needed
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
    p_logreg <- predict(logreg_model, new_user, type = "response")
    p_knn    <- predict(knn_model,    new_user, type = "prob")[,2]
    p_ann    <- predict(ann_model,    new_user, type = "raw") # May need [,1] depending on nnet setup
    p_svm    <- predict(svm_model,    new_user, type = "probabilities")[,2]
    p_tree   <- predict(tree_model,   new_user, type = "prob")[,2]
    
    # 2. Build the Level 1 Stack (Columns must match your meta-model training exactly)
    live_stack <- data.frame(
      logreg = as.numeric(p_logreg),
      knn    = as.numeric(p_knn),
      ann    = as.numeric(p_ann),
      svm    = as.numeric(p_svm),
      tree   = as.numeric(p_tree)
    )
    
    # 3. Final Meta-Prediction
    final_pred <- predict(meta_model, live_stack)
    
    list(final = final_pred, probs = live_stack)
  })
  
  output$final_result <- renderText({
    res <- prediction_data()$final
    if(res == "1") "Likely Depressed" else "Not Likely Depressed"
  })
  
  output$expert_probs <- renderTable({
    prediction_data()$probs
  })
}

shinyApp(ui = ui, server = server)