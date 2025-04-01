
# adjust WD
setwd("./..")

# source R scripts and Python list of tasks
source("r/eval.R")
source("r/eval-helpers.R")
source("r/helpers.R")
source("r/zzz-deps.R")
source_python("py/task_spec.py")

# Models and Tasks
models <- c("llama3_8b_instruct", "llama3_70b_instruct")

tasks <- py$task_specs
tasks <- tasks[1:7]

df <- expand.grid(model = models, task = seq_along(tasks), stringsAsFactors = FALSE)
df$score <- df$eval <- NA
for (i in seq_len(nrow(df))) {
  
  df$eval[i] <- list(eval_task(df$model[i], tasks[[df$task[i]]]))
  df$score[i] <- eval_to_score(df$eval[i][[1]])
}

ui <- fluidPage(
  theme = shinytheme("cosmo"),
  titlePanel("Benchmark Results"),
  
  div(style = "display: flex; justify-content: center;", 
      plotOutput("barplot", height = "600px", width = "900px")
  ),
  DTOutput("score_table"),
  div(style = "display: flex; justify-content: center;", 
      plotOutput("score_plot", height = "600px", width = "900px")
  )
)

server <- function(input, output, session) {
  output$barplot <- renderPlot({
    
    scboard <- as.data.table(df)[, list(score = sum(score)), by = "model"]
    scboard[, Model := model_name(model)]
    ggplot(scboard, aes(x = Model, y = score, fill = Model)) +
      geom_col() +
      geom_text(aes(label = round(score)), vjust = -0.5) +
      geom_hline(yintercept = 100 * length(tasks), color = "darkgreen") +
      annotate("text", x = 1.5, y = 100 * length(tasks) - 5, label = "Perfect Score", color = "darkgreen", fontface = "bold") +
      theme_minimal() +
      labs(title = "Total Score by Model", y = "Score", x = "Model")
  }, res = 120)
  
  output$score_table <- renderDT({
    sctask <- df
    sctask$score <- round(sctask$score)
    sctask$model <- model_name(sctask$model)
    sctask <- reshape2::dcast(sctask, task ~ model, value.var = "score")
    sctask$task <- sapply(tasks, `[[`, "name")[sctask$task]
    datatable(
      sctask,
      options = list(
        pageLength = Inf,
        dom = "t",
        columnDefs = list(list(className = "dt-center", targets = "_all"))
      ),
      escape = FALSE,  # Allow HTML rendering
      callback = JS("table.on('click', 'td', function() {
  var rowIdx = table.cell(this).index().row;
  var colIdx = table.cell(this).index().column;
  var score = $(this).text();
  var model = table.column(colIdx).header().innerText;
  var taskIndex = rowIdx + 1;  // Convert to 1-based index

  Shiny.setInputValue('sel_model', model, {priority: 'event'});
  Shiny.setInputValue('task_idx', taskIndex, {priority: 'event'});
});
")
    )
  })
  
  output$score_plot <- renderPlot({
    req(input$sel_model, input$task_idx)
    model_name <- model_unname(input$sel_model)
    df_idx <- which(df$task == input$task_idx & df$model == model_name)
    plt_eval(df$eval[[df_idx]], model_name, 
             tasks[[input$task_idx]])
  }, res = 120)
}

shinyApp(ui, server)
