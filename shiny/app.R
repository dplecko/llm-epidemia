# adjust WD
setwd("./..")

# source R scripts and Python list of tasks
source("r/eval.R")
source("r/eval-helpers.R")
source("r/helpers.R")
source("r/zzz-deps.R")
source_python("workspace/task_spec.py")
sync_bench() # download the data from the server

# Load fonts and setup theme
library(sysfonts)
library(showtext)
library(bslib)
font_add_google("Inter", "inter")
showtext_auto()

# ----- DESIGN SETTINGS -----
# Control appearance from here
font_family <- "inter"
dark_theme <- FALSE
plot_dpi_main <- 200
plot_dpi_detail <- 120
plot_width <- "100%"
plot_max_width <- "700px"

# UI CSS
custom_css <- if (dark_theme) {
  HTML(sprintf("body { background-color: #121212; color: #e0e0e0; font-family: '%s'; }\n.card { background-color: #1e1e1e; box-shadow: 0 0 15px rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; }\nh2, h3 { font-weight: 700; text-align: center; color: #ffffff; }", font_family))
} else {
  HTML(sprintf("body { background-color: #f8f9fa; color: #212529; font-family: '%s'; }\n.card { background-color: white; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 12px; padding: 20px; }\nh2, h3 { font-weight: 700; text-align: center; color: #212529; }", font_family))
}

theme_set(theme_minimal(base_family = font_family))

# Models and Tasks
d2d <- FALSE
if (d2d) {
  models <- c("nhanes", "gss")
  task_sel <- c(1, 2)
} else {
  models <- c("llama3_8b_instruct", "mistral_7b_instruct", "phi4",
              "gemma3_27b_instruct", "llama3_70b_instruct")
  # models <- c("llama3_8b_instruct", "mistral_7b_instruct" , 
  #             "phi4", "gemma3_27b_instruct", "llama3_70b_instruct")
  task_sel <- TRUE
}

tasks <- py$task_specs
tasks <- tasks[task_sel]

df <- expand.grid(model = models, task = seq_along(tasks), stringsAsFactors = FALSE)
df$score <- df$eval <- NA
for (i in seq_len(nrow(df))) {
  df$eval[i] <- list(eval_task(df$model[i], tasks[[df$task[i]]]))
  df$score[i] <- eval_to_score(df$eval[i][[1]])
}

ui <- fluidPage(
  theme = bs_theme(bootswatch = "darkly", version = 5),
  # tags$head(tags$style(custom_css)),
  titlePanel("L1 Faithfulness Benchmark"),
  tags$style(HTML("
  table.dataTable tbody td.selected-cell {
    background-color: #1a73e8 !important;
    color: white !important;
  }
  table.dataTable tbody tr.selected > * {
    background-color: inherit !important;
    color: inherit !important;
  }
  ")),
  div(class = "card", style = paste0("margin: auto; max-width: ", plot_max_width, ";"),
      plotOutput("barplot", height = "400px", width = plot_width)),
  
  div(class = "card", DTOutput("score_table")),
  
  div(class = "card", style = paste0("margin: auto; max-width: ", plot_max_width, ";"),
      plotOutput("score_plot", height = "400px", width = plot_width))
)

server <- function(input, output, session) {
  output$barplot <- renderPlot({
    scboard <- as.data.table(df)[, list(score = sum(score)), by = "model"]
    scboard[, Model := model_name(model)]
    scboard <- setorder(scboard, -score)
    scboard[, Model := factor(Model, levels = unique(Model))]
    ggplot(scboard, aes(x = Model, y = score, fill = Model)) +
      geom_col(width = 0.6, show.legend = FALSE) +
      geom_text(aes(label = round(score)), vjust = -0.5, color = "darkred", size = 5, fontface = "bold") +
      geom_hline(yintercept = 100 * length(tasks), color = "darkgreen", linetype = "dashed", linewidth = 0.6) +
      annotate("text", x = 1.5, y = 100 * length(tasks) - 50, label = "Perfect Score", color = "darkgreen", fontface = "bold") +
      scale_fill_viridis_d() +
      labs(title = "Total Score by Model", y = "Score", x = "Model") +
      theme_bw(base_size = 16, base_family = font_family) +
      theme(
        plot.title = element_text(face = "bold", hjust = 0.5, size = 20, color = if (dark_theme) "white" else "black"),
        axis.title = element_text(face = "bold", color = if (dark_theme) "white" else "black"),
        axis.text = element_text(color = if (dark_theme) "white" else "black"),
        panel.grid.major.y = element_line(color = if (dark_theme) "darkgray" else "gray85"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = if (dark_theme) "#1e1e1e" else "white", color = NA),
        panel.background = element_rect(fill = if (dark_theme) "#1e1e1e" else "white", color = NA),
        axis.line = element_blank()
      )
  }, res = plot_dpi_main)
  
  output$score_table <- renderDT({
    sctask <- df
    sctask$score <- round(sctask$score)
    sctask$model <- model_name(sctask$model)
    sctask <- reshape2::dcast(sctask, task ~ model, value.var = "score")
    sctask$task <- sapply(tasks, `[[`, "name")[sctask$task]
    datatable(
      sctask,
      options = list(
        pageLength = 100,
        dom = "t",
        columnDefs = list(list(className = "dt-center", targets = "_all"))
      ),
      escape = FALSE,
      selection = 'none',
      callback = JS("
  table.on('click', 'td', function() {
    table.$('td').removeClass('selected-cell'); // remove previous selection
    $(this).addClass('selected-cell'); // highlight current cell

    var rowIdx = table.cell(this).index().row;
    var colIdx = table.cell(this).index().column;
    var score = $(this).text();
    var model = table.column(colIdx).header().innerText;
    var taskIndex = rowIdx + 1;

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
    plt_eval(df$eval[[df_idx]], model_name, tasks[[input$task_idx]])
  }, res = plot_dpi_detail)
}

shinyApp(ui, server)